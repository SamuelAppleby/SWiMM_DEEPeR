using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimulationManager : MonoBehaviour
{
    private static SimulationManager _instance;
    public struct ServerInfo
    {
        public string URL;
        public int Port;
        public int TickRate;
    }

    public ServerInfo server;
    public bool useServer = false;

    public static SimulationManager Instance
    {
        get
        {
            if (_instance == null)
            {
                Debug.LogError("NO GAME MANAGER");
            }

            return _instance;
        }
    }

    void Start()
    {

    }

    void Update()
    {

    }

    private void ParseCommandLineArguments(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "server":
                    if (i < args.Length)
                    {
                        string[] parts = args[++i].Split(':');

                        if (parts.Length == 2)
                        {
                            server.URL = parts[0];
                            server.Port = Int32.Parse(parts[1]);
                            server.TickRate = 1;
                            useServer = true;
                        }
                    }
                    break;
            }
        }
    }

    private void Awake()
    {
        _instance = this;
        ParseCommandLineArguments(System.Environment.GetCommandLineArgs());
    }
}
