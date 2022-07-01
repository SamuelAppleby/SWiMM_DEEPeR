using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.SceneManagement;
using static Server;

public class SimulationManager : Singleton<SimulationManager>
{
    [HideInInspector]
    public bool in_manual_mode;

    private string network_config_dir;
    private string debug_config_dir;

    [HideInInspector]
    public Server server;

    [HideInInspector]
    public GameObject rover;

    [HideInInspector]
    public int avgFrameRate;

    private FullScreenMode[] screenmodes;
    private int screenIndex;

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
    private string main_menu_name = "MainMenu";

    public bool IsInitialized { get; private set; }

    [HideInInspector]
    public GameObject[] lighting_objs;

    [HideInInspector]
    public GameObject[] water_objs;

    private void Start()
    {
        _instance.InvokeRepeating("UpdateFPS", 1, 1);

#if UNITY_EDITOR
        _instance.debug_config_dir = "../Configs/data/client_debug_config.json";
        _instance.network_config_dir = "../Configs/data/network_config.json";
#else
        _instance.debug_config_dir = "../../../Configs/data/client_debug_config.json";
        _instance.network_config_dir = "../../../Configs/data/network_config.json";
#endif

        _instance.ProcessConfig(ref _instance.debug_config, _instance.debug_config_dir);
        _instance.ProcessConfig(ref _instance.network_config, _instance.network_config_dir);
        _instance.PurgeAndCreateDirectory(_instance.debug_config.payload.packet_sent_dir);
        _instance.PurgeAndCreateDirectory(_instance.debug_config.payload.image_dir);

        _instance.screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];

        _instance.server = new Server();
        _instance.in_manual_mode = true;

        _instance.ParseCommandLineArguments(System.Environment.GetCommandLineArgs());

        _instance.IsInitialized = true;
    }

    private void PurgeAndCreateDirectory(string dir_path)
    {
        if (Directory.Exists(dir_path))
        {
            Directory.Delete(dir_path, true);
        }

        Directory.CreateDirectory(dir_path);
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

    protected override void Awake()
    {
        base.Awake();

        _instance.water_objs = GameObject.FindGameObjectsWithTag("Water");
        _instance.lighting_objs = GameObject.FindGameObjectsWithTag("Lighting");
    }

    protected override void OnSceneChanged()
    {
        base.OnSceneChanged();

        if (SceneManager.GetActiveScene().name == "UnderwaterScene")
        {
            if (_instance.server.server_config.is_overridden)
            {
                RenderSettings.fogMode = FogMode.Exponential;
                RenderSettings.fog = _instance.server.server_config.payload.envConfig.fogConfig.fogOn;
                RenderSettings.fogDensity = _instance.server.server_config.payload.envConfig.fogConfig.fogDensity;
                RenderSettings.fogColor = new Color(_instance.server.server_config.payload.envConfig.fogConfig.fogColour[0], 
                    _instance.server.server_config.payload.envConfig.fogConfig.fogColour[1], _instance.server.server_config.payload.envConfig.fogConfig.fogColour[2]);
            }
        }
    }

    void Update()
    {
        globalControls.Update(_instance.in_manual_mode);

        if (_instance.server.server_crash)
        {
            SceneManager.LoadScene(_instance.main_menu_name);
            _instance.server.server_crash = false;
        }

        if (globalControls.quitting)
        {
            if(SceneManager.GetActiveScene().name != _instance.main_menu_name)
            {
                SceneManager.LoadScene(_instance.main_menu_name);
            }
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


        if (globalControls.cursor_change)
        {
            Cursor.lockState = Cursor.lockState == CursorLockMode.Locked ? CursorLockMode.None : CursorLockMode.Locked;
            Cursor.visible = !Cursor.visible;
        }

        if (globalControls.water_toggle)
        {
            if (_instance.water_objs.Length > 0)
            {
                foreach(GameObject obj in _instance.water_objs)
                {
                    obj.SetActive(!obj.activeSelf);
                }
            }
        }

        if (globalControls.volumetric_lighting_toggle)
        {
            if (_instance.lighting_objs.Length > 0)
            {
                foreach (GameObject obj in _instance.lighting_objs)
                {
                    obj.SetActive(!obj.activeSelf);
                }
            }
        }

        if (globalControls.reset_ncps)
        {
            foreach (Transform group_trans in FishSpawner.m_group_parent.GetComponentInChildren<Transform>().transform)
            {
                foreach (Transform ai in group_trans.transform)
                {
                    Destroy(ai.gameObject);
                }
            }
        }
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
