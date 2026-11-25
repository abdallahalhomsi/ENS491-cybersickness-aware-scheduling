#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/flow-monitor-module.h"
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include "ns3/bursty-helper.h"
using namespace ns3;

// Helper function to calculate the Risk-Aware Priority
double GetPriority(double R_i, double D_i_minus_t, double alpha) {
    double epsilon = 0.0001;
    double deadline_term = std::max(D_i_minus_t, epsilon);
    return (1.0 + alpha * R_i) / deadline_term;
}

int main(int argc, char* argv[]) {
    double simTime = 20.0;
    uint32_t numUsers = 3;
    
    // Default EDF (alpha = 0). Change to 1.0 for Risk-Aware.
    double alpha = 1.0; 
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("alpha", "Risk weight (0 for EDF, >0 for Risk-Aware)", alpha);
    cmd.Parse(argc, argv);

    // 1. Read ML Risk Scores
    std::map<uint32_t, double> riskScores;
    std::ifstream file("ml_risk_scores.json");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("user_") != std::string::npos) {
                size_t colon = line.find(":");
                size_t quote = line.find("\"");
                if (colon != std::string::npos && quote != std::string::npos) {
                    // Manual parse: "user_1": 0.85,
                    std::string numStr = line.substr(quote + 6, line.find("\"", quote + 1) - (quote + 6));
                    std::string scoreStr = line.substr(colon + 1, line.find(",") - (colon + 1));
                    riskScores[std::stoi(numStr)] = std::stod(scoreStr);
                }
            }
        }
    }
    std::cout << "--- Running Simulation with Alpha = " << alpha << " ---" << std::endl;
    for(auto const& [id, score] : riskScores) {
        std::cout << "User " << id << " Risk: " << score << std::endl;
    }

    // 2. Network Setup
    NodeContainer users;
    users.Create(numUsers);
    NodeContainer router;
    router.Create(1);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("10ms"));

    std::vector<NetDeviceContainer> devices;
    for (uint32_t i = 0; i < numUsers; i++) {
        devices.push_back(p2p.Install(users.Get(i), router.Get(0)));
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

    // 3. Apply EDF / Risk-Aware Queue
    TrafficControlHelper tch;
    for (uint32_t i = 0; i < numUsers; i++) {
        tch.Uninstall(devices[i].Get(1)); // Remove default
    }
    // Set the queue disc type only ONCE
    tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc"); 
    for (uint32_t i = 0; i < numUsers; i++) {
        tch.Install(devices[i].Get(1)); 
    }

    // 4. Generate Traffic
    uint16_t port = 50000;
    for (uint32_t i = 0; i < numUsers; i++) {
        Address sinkAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
        PacketSinkHelper sink("ns3::UdpSocketFactory", sinkAddress);
        ApplicationContainer sinkApps = sink.Install(users.Get(i));
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simTime));

        BurstyHelper burstyHelper("ns3::UdpSocketFactory",
                                  InetSocketAddress(users.Get(i)->GetObject<Ipv4>()->GetAddress(1,0).GetLocal(), port));
        burstyHelper.SetAttribute("FragmentSize", UintegerValue(1200));
        
        // We use the built-in VR generator so it reads the trace file we put in the build folder
        burstyHelper.SetBurstGenerator("ns3::VrBurstGenerator",
                                       "FrameRate", DoubleValue(60.0), 
                                       "TargetDataRate", DataRateValue(DataRate("100Mbps")),
                                       "VrAppName", StringValue("VirusPopper")); 

        ApplicationContainer serverApps = burstyHelper.Install(router.Get(0));
        serverApps.Start(Seconds(1.0));
        serverApps.Stop(Seconds(simTime));
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    std::cout << "\n--- FINAL QUEUE STATS ---" << std::endl;
    for (uint32_t i = 0; i < numUsers; i++) {
        // This prints the risk score for each user so we know who got priority
        std::cout << "User " << (i+1) << " Risk: " << riskScores[i+1] 
                  << " (Scheduler Priority weight: " << (1.0 + alpha * riskScores[i+1]) << ")" << std::endl;
    }

    // --- PRINT FINAL METRICS ---
    std::cout << "\n--- SCHEDULER PRIORITY WEIGHTS ---" << std::endl;
    for (uint32_t i = 1; i <= numUsers; i++) {
        double risk = riskScores[i];
        double priority_weight = 1.0 + (alpha * risk);
        std::cout << "User " << i << " | Risk: " << risk 
                  << " | Priority Weight (1 + alpha * R): " << priority_weight << std::endl;
    }

    Simulator::Destroy();


    std::cout << "Simulation finished successfully." << std::endl;
    std::cout << "Check FlowMonitor or pcap for detailed drop stats." << std::endl;
    return 0;
}
