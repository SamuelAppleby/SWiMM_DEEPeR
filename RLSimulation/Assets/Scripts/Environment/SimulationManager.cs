using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.SceneManagement;
using static Server;

public class SimulationManager : Singleton<SimulationManager>
{
    public bool in_manual_mode = false;

    private string network_config_dir;
    private string debug_config_dir;

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
        public bool save_packet_data;
        public bool server_set;
    }

    public struct CommandLineArguemnts
    {
        public ServerInfo server_info;
    }

    private CommandLineArguemnts command_args;

    [Serializable]
    public struct DebugConfig
    {
        public bool save_images;
        public string image_dir;
        public bool save_sent_packets;
        public string packet_sent_dir;
    }

    /* Local configs for reading */
    [HideInInspector]
    public JsonMessage<DebugConfig> debug_config;


    [Serializable]
    public struct NetworkConfig
    {
        public string host;
        public int port;
        public Buffers buffers;
    }

    [HideInInspector]
    public JsonMessage<NetworkConfig> network_config;

    [SerializeField]
    private string main_menu_name;

    public bool IsInitialized { get; private set; }

    [SerializeField]
    private MainMenu main_menu;

    private void Start()
    {
#if UNITY_EDITOR
        debug_config_dir = "../Configs/data/client_debug_config.json";
        network_config_dir = "../Configs/data/network_config.json";
#else
        debug_config_dir = "../../../Configs/data/client_debug_config.json";
        network_config_dir = "../../../Configs/data/network_config.json";
#endif

        ProcessConfig(ref debug_config, debug_config_dir);
        ProcessConfig(ref network_config, network_config_dir);

        _instance.screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];

        _instance.server = new Server();

        ////ParseCommandLineArguments(System.Environment.GetCommandLineArgs());

        //// If not set via command line (normal flow)
        //if (command_args.server_info.server_set)
        //{
        //    _instance.server.ip = command_args.server_info.ip;
        //    _instance.server.port = command_args.server_info.port;
        //}

        IsInitialized = true;
    }

    public async Task<Exception> ConnectToServer(string ip, int port)
    {
        network_config.payload.host = ip;
        network_config.payload.port = port;
        return await server.Connect(network_config.payload);
    }

    public void OnServerDisconnected()
    {
        Debug.Log("No server response");
        SceneManager.LoadScene(main_menu_name);
    }

    void Update()
    {
        globalControls.Update(in_manual_mode);

        if (_instance.server.server_crash || globalControls.quitting)
        {
            SceneManager.LoadScene(main_menu_name);
            _instance.server.server_crash = false;
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


    public void Clear_Cache()
    {
        if (debug_config.is_overridden)
        {
            DirectoryInfo di = new DirectoryInfo(debug_config.payload.packet_sent_dir);

            if (di.Exists)
            {
                FileInfo[] files = di.GetFiles();
                foreach (FileInfo file in files)
                {
                    file.Delete();
                }
            }
            else
            {
                System.IO.Directory.CreateDirectory(di.FullName);
            }

            di = new DirectoryInfo(debug_config.payload.image_dir);

            if (di.Exists)
            {
                FileInfo[] files = di.GetFiles();
                foreach (FileInfo file in files)
                {
                    file.Delete();
                }
            }
            else
            {
                System.IO.Directory.CreateDirectory(di.FullName);
            }
        }
    }

    public void ProcessConfig<T>(ref JsonMessage<T> config, string dir)
    {
        using (StreamReader r = new StreamReader(dir))
        {
            string json = r.ReadToEnd();
            config.payload = JsonUtility.FromJson<T>(json);
            config.is_overridden = true;
        }
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
