using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using static Server;

public class SimulationManager : Singleton<SimulationManager>
{
    public bool manual_server = false;
    public string ip_addr;
    public int port;

    [HideInInspector]
    public Server server;

    [HideInInspector]
    public GameObject rover;

    private int avgFrameRate;

    private FullScreenMode[] screenmodes;
    private int screenIndex = 0;

    static public GlobalControlSettings globalControls = new GlobalControlSettings();

    public struct ServerInfo
    {
        public string ip;
        public int port;
        public bool server_set;
    }

    public struct CommandLineArguemnts
    {
        public ServerInfo server_info;
    }

    private CommandLineArguemnts command_args;

    async void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        _instance.screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];

        ParseCommandLineArguments(System.Environment.GetCommandLineArgs());

        if (manual_server)
        {
            command_args.server_info.port = port;
            command_args.server_info.ip = ip_addr;
            command_args.server_info.server_set = true;
        }

        if (command_args.server_info.server_set)
        {
            _instance.server = new Server(command_args.server_info);
            await server.Connect();
            server.ContinueRead();
        }
    }

    void Update()
    {
        globalControls.Update(server != null && server.IsTcpGood());

        if (globalControls.quitting)
        {
            TerminateApplication();
        }

        if (globalControls.reload_scene)
        {
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            globalControls.reload_scene = false;
        }

        if (globalControls.changeWindow)
        {
            _instance.screenIndex = _instance.screenIndex == screenmodes.Length - 1 ? 0 : screenIndex + 1;
            Screen.fullScreenMode = _instance.screenmodes[screenIndex];
        }

        _instance.UpdateFPS();
    }

    private void TerminateApplication()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
         Application.Quit();
#endif
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
                            command_args.server_info.ip = parts[0];
                            command_args.server_info.port = Int32.Parse(parts[1]);
                            command_args.server_info.server_set = true;
                        }
                    }
                    break;
            }
        }
    }

    private void UpdateFPS()
    {
        float current = (int)(1f / Time.unscaledDeltaTime);
        _instance.avgFrameRate = (int)current;
    }
}
