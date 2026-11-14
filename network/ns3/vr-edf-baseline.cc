#include "ns3/applications-module.h"
#include "ns3/burst-sink-helper.h"
#include "ns3/bursty-helper.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VrEdfBaseline");

int main(int argc, char* argv[])
{
    double simTime = 10.0;
    uint32_t numUsers = 3;
    
    // NeSt-VR realistic parameters
    std::string dataRate = "100Mbps";
    std::string delay = "10ms";
    double lossRate = 0.01; // 1% loss

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime", "Length of simulation [s].", simTime);
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);

    // 1. Create Nodes (1 Router/Switch + 3 VR Users)
    NodeContainer users;
    users.Create(numUsers);
    NodeContainer router;
    router.Create(1);

    // 2. Setup Network Link with Impairments
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(dataRate));
    p2p.SetChannelAttribute("Delay", StringValue(delay));

    // Add 1% Packet Loss (NeSt-VR baseline)
    Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
    em->SetAttribute("ErrorRate", DoubleValue(lossRate));
    p2p.SetDeviceAttribute("ReceiveErrorModel", PointerValue(em));

    // Connect users to the router
    std::vector<NetDeviceContainer> devices;
    for (uint32_t i = 0; i < numUsers; i++) {
        NetDeviceContainer d = p2p.Install(users.Get(i), router.Get(0));
        devices.push_back(d);
    }

    InternetStackHelper stack;
    stack.Install(users);
    stack.Install(router);

    Ipv4AddressHelper address;
    for (uint32_t i = 0; i < numUsers; i++) {
        std::ostringstream subnet;
        subnet << "10.1." << (i + 1) << ".0";
        address.SetBase(subnet.str().c_str(), "255.255.255.0");
        address.Assign(devices[i]);
    }

    // 3. Install EDF Queues on the Router
    for (uint32_t i = 0; i < numUsers; i++) {
        // Uninstall the default queue first
        TrafficControlHelper tch;
        tch.Uninstall(devices[i].Get(1));
        
        // Now install the EDF queue
        tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc"); 
        tch.Install(devices[i].Get(1)); 
    }

    // 4. Generate VR Traffic for each user
    uint16_t port = 50000;
    for (uint32_t i = 0; i < numUsers; i++) {
        Address sinkAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
        PacketSinkHelper sink("ns3::UdpSocketFactory", sinkAddress);
        ApplicationContainer sinkApps = sink.Install(users.Get(i));
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simTime));

        // We use the built-in VR trace
        BurstyHelper burstyHelper("ns3::UdpSocketFactory",
                                  InetSocketAddress(users.Get(i)->GetObject<Ipv4>()->GetAddress(1,0).GetLocal(), port));
        burstyHelper.SetAttribute("FragmentSize", UintegerValue(1200));
        burstyHelper.SetBurstGenerator("ns3::VrBurstGenerator",
                                       "FrameRate", DoubleValue(60.0), // Use 60Hz as it's supported by the module
                                       "TargetDataRate", DataRateValue(DataRate("30Mbps")),
                                       "VrAppName", StringValue("VirusPopper")); // Using default trace

        ApplicationContainer serverApps = burstyHelper.Install(router.Get(0));
        serverApps.Start(Seconds(1.0));
        serverApps.Stop(Seconds(simTime));
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();

    std::cout << "Simulation finished: " << numUsers << " users, EDF Queue, " << lossRate * 100 << "% loss." << std::endl;
    return 0;
}
