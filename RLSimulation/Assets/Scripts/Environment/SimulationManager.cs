using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using static Server;
using AsyncOperation = UnityEngine.AsyncOperation;
using Random = UnityEngine.Random;

public class SimulationManager : Singleton<SimulationManager>
{
    [HideInInspector]
    public SceneIndices current_scene_index;
    public TextMeshProUGUI tips_text;
    public CanvasGroup alpha_canvas;
    public string[] tips;
    public Sprite[] background_images;
    public Image background_image;

    private AsyncOperation scene_loading;

    private float total_scene_progress;
    private float total_spawn_progress;

    public TextMeshProUGUI text_field;
    public GameObject loading_screen;
    public Slider progress_bar;

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

    public GameObject processing_obj;

    public struct ServerInfo
    {
        public string ip;
        public int port;
        public bool save_packet_data;
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
    public DebugConfig debug_config;

    Dictionary<string, Protocol> protocol_mapping = new Dictionary<string, Protocol>
    {
        { "udp", Protocol.UDP },
        { "tcp", Protocol.TCP }
    };

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct NetworkConfig
    {
        public Protocol e_protocol;
        public string protocol;
        public string host;
        public int port;
        public Buffers buffers;
    }

    [HideInInspector]
    public NetworkConfig network_config;

    public bool IsInitialized { get; private set; }

    [HideInInspector]
    public GameObject[] lighting_objs;

    [HideInInspector]
    public GameObject[] water_objs;

    public void OnObservationSent()
    {
        _instance.server.observations_sent++;

        if (_instance.server.json_server_config.payload.serverConfig.envConfig.freezeOnTelemetry)
        {
            Time.timeScale = 0;
        }
    }

    public void OnActionReceived(JsonMessage param)
    {
        _instance.server.actions_received++;

        if (_instance.server.json_server_config.payload.serverConfig.envConfig.freezeOnTelemetry)
        {
            Time.timeScale = 1;
        }
    }

    public async void OnServerConnectionResponse(Exception e)
    {
        processing_obj.SetActive(e == null);
        processing_obj.GetComponentInChildren<TextMeshProUGUI>().text = "Model initialising...";

        await Task.Run(() => server.ContinueReadWrite());
        // TODO Clean up server caches
    }

    public void OnServerConfigReceived(JsonMessage param)
    {
        _instance.server.json_server_config = param;

        if (_instance.server != null && _instance.server.IsTcpGood())
        {
            _instance.server.obsv = new DataToSend
            {
                msg_type = "server_config_received",
                payload = new Telemetary_Data { }
            };
        }
    }

    public void EpisodeReset(bool in_manual)
    {
        _instance.server.resets_received++;

        switch (current_scene_index)
        {
            case SceneIndices.MAIN_MENU:
                processing_obj.SetActive(false);
                break;
            case SceneIndices.SIMULATION:
                MoveToScene(SceneIndices.SIMULATION, in_manual);       // In this case will unload and reload as intended
                break;
        }
    }

    public void ExitCurrentScene()
    {
        MoveToScene(SceneIndices.SIMULATION);       // In this case will unload and reload as intended
    }

    public void EndSimulation()
    {
        MoveToScene(SceneIndices.EXIT);       // As above
    }

    private void Start()
    {
        _instance.InvokeRepeating("UpdateFPS", 0f, 1);

#if UNITY_EDITOR
        _instance.debug_config_dir = "../Configs/data/client_debug_config.json";
        _instance.network_config_dir = "../Configs/data/network_config.json";
#else
        _instance.debug_config_dir = "../../../Configs/data/client_debug_config.json";
        _instance.network_config_dir = "../../../Configs/data/network_config.json";
#endif

        _instance.ParseCommandLineArguments(Environment.GetCommandLineArgs());

        _instance.ProcessConfig(ref _instance.debug_config, _instance.debug_config_dir);
        _instance.ProcessConfig(ref _instance.network_config, _instance.network_config_dir);

        _instance.network_config.e_protocol = protocol_mapping[_instance.network_config.protocol];

        _instance.PurgeAndCreateDirectory(_instance.debug_config.packet_sent_dir);
        _instance.PurgeAndCreateDirectory(_instance.debug_config.image_dir);

        _instance.screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];

        _instance.server = null;
        _instance.in_manual_mode = true;

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

    public void OnROVInitialised(GameObject rov)
    {
        rover = rov;
    }

    public async void ConnectToServer(string ip, int port)
    {
        server = new Server();
        Exception e = await server.Connect(network_config, ip, port);
        EventMaster._instance.server_connection_attempt_event.Raise(e);
    }

    public void OnServerDisconnected()
    {
        Debug.Log("No server response");
        SceneManager.LoadScene((int)SceneIndices.MAIN_MENU);
    }

    protected override void Awake()
    {
        base.Awake();
        _instance.MoveToScene(SceneIndices.MAIN_MENU);
        _instance.water_objs = GameObject.FindGameObjectsWithTag("Water");
        _instance.lighting_objs = GameObject.FindGameObjectsWithTag("Lighting");
    }

    public void MoveToScene(SceneIndices to, bool in_manual = false)
    {
        if (to == SceneIndices.EXIT)
        {
            _instance.QuitApplication();
            return;
        }

        _instance.background_image.sprite = background_images[Random.Range(0, background_images.Length)];
        _instance.loading_screen.gameObject.SetActive(true);

        _instance.in_manual_mode = in_manual;

        /* We don't want to unload the persistent scene */
        if (_instance.current_scene_index == SceneIndices.PERSISTENT_SCENE)
        {
            StartCoroutine(GenerateTips());
            StartCoroutine(GetSceneLoadProgress());

            _instance.scene_loading = SceneManager.LoadSceneAsync((int)to, LoadSceneMode.Additive);

            _instance.scene_loading.completed += handle =>
            {
                OnSceneChanged(handle, to);
            };
        }

        else
        {
            SceneManager.UnloadSceneAsync((int)_instance.current_scene_index).completed += handle =>
            {
                StartCoroutine(GenerateTips());
                StartCoroutine(GetSceneLoadProgress());

                _instance.scene_loading = SceneManager.LoadSceneAsync((int)to, LoadSceneMode.Additive);

                _instance.scene_loading.completed += handle =>
                {
                    OnSceneChanged(handle, to);
                };

            };
        }
    }

    public IEnumerator GenerateTips()
    {
        int tip_count = Random.Range(0, tips.Length);
        tips_text.text = tips[tip_count];

        while (loading_screen.activeInHierarchy)
        {
            yield return new WaitForSeconds(3f);

            alpha_canvas.alpha = Mathf.Lerp(alpha_canvas.alpha, 0, 0.1f);

            yield return new WaitForSeconds(.5f);

            tip_count++;

            if (tip_count >= tips.Length)
            {
                tip_count = 0;
            }

            tips_text.text = tips[tip_count];

            alpha_canvas.alpha = Mathf.Lerp(alpha_canvas.alpha, 1, 0.1f);
        }
    }

    public IEnumerator GetSceneLoadProgress()
    {
        while (_instance.scene_loading != null && !_instance.scene_loading.isDone)
        {
            _instance.total_scene_progress = _instance.scene_loading.progress * 100f;
            _instance.text_field.text = string.Format("Loading Scene: {0}%", _instance.total_scene_progress);

            yield return null;
        }

        while (FishSpawner.current != null && !FishSpawner.current.is_done)
        {
            _instance.total_spawn_progress = Mathf.Round(FishSpawner.current.current_progress * 100f);
            switch (FishSpawner.current.current_stage)
            {
                case InitialisationStage.INITIALISING_NPCS:
                    _instance.text_field.text = string.Format("Initialising NPCs {0}%", _instance.total_spawn_progress);
                    break;
                case InitialisationStage.SPAWNING_NPCS:
                    _instance.text_field.text = string.Format("Spawning NPCs {0}%", _instance.total_spawn_progress);
                    break;
            }
        }

        _instance.progress_bar.value = Mathf.RoundToInt(Mathf.Round((_instance.total_scene_progress + _instance.total_spawn_progress) / 2f));
        _instance.loading_screen.gameObject.SetActive(false);

        yield break;
    }

    protected override void OnSceneChanged(AsyncOperation handle, SceneIndices to)
    {
        base.OnSceneChanged(handle, to);
        _instance.scene_loading = null;
        _instance.current_scene_index = to;
    }

    public void QuitApplication()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
         Application.Quit();
#endif
    }

    public void IndexWindow()
    {
        _instance.screenIndex = _instance.screenIndex == screenmodes.Length - 1 ? 0 : screenIndex + 1;
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];
    }

    public void IndexCursor()
    {
        Cursor.lockState = Cursor.lockState == CursorLockMode.Locked ? CursorLockMode.None : CursorLockMode.Locked;
        Cursor.visible = !Cursor.visible;
    }

    public void ToggleWater()
    {
        if (_instance.water_objs.Length > 0)
        {
            foreach (GameObject obj in _instance.water_objs)
            {
                obj.SetActive(!obj.activeSelf);
            }
        }
    }

    public void ToggleVolumetricLighting()
    {
        if (_instance.lighting_objs.Length > 0)
        {
            foreach (GameObject obj in _instance.lighting_objs)
            {
                obj.SetActive(!obj.activeSelf);
            }
        }
    }

    public void ResetNPCs()
    {
        foreach (Transform group_trans in FishSpawner.m_group_parent.GetComponentInChildren<Transform>().transform)
        {
            foreach (Transform ai in group_trans.transform)
            {
                Destroy(ai.gameObject);
            }
        }
    }

    public void MonitorAndFireServerEvents()
    {
        if (_instance.server.json_server_config.is_overriden)
        {
            EventMaster._instance.server_config_received_event.Raise(_instance.server.json_server_config);
            _instance.server.json_server_config.is_overriden = false;
        }

        if (_instance.server.json_reset_episode.is_overriden)
        {
            EventMaster._instance.reset_episode_event.Raise();
            _instance.server.json_reset_episode.is_overriden = false;
        }

        if (_instance.server.json_rover_controls.is_overriden)
        {
            EventMaster._instance.json_control_event.Raise(_instance.server.json_rover_controls);
            _instance.server.json_rover_controls.is_overriden = false;
        }

        if (_instance.server.json_end_simulation.is_overriden)
        {
            EventMaster._instance.end_simulation_event.Raise();
            _instance.server.json_end_simulation.is_overriden = false;
        }

        if (_instance.server.last_obsv != null)
        {
            switch (_instance.server.last_obsv.Value.msg_type)
            {
                case "on_server_config_received":
                    break;
                case "on_telemetry":
                    EventMaster._instance.observation_sent_event.Raise();
                    break;
                default:
                    break;
            }

            _instance.server.last_obsv = null;
        }
    }

    void Update()
    {
        globalControls.Update(_instance.in_manual_mode);

        if (_instance.server != null && _instance.server.server_crash)
        {
            SceneIndices moving_to = current_scene_index == SceneIndices.MAIN_MENU ? SceneIndices.EXIT : SceneIndices.MAIN_MENU;
            MoveToScene(moving_to);

            if (_instance.server.server_crash)
            {
                _instance.server.server_crash = false;
            }
        }

        if (_instance.server != null)
        {
            MonitorAndFireServerEvents();
        }
    }

    public void Clear_Cache()
    {
        DirectoryInfo di = new DirectoryInfo(debug_config.packet_sent_dir);

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

        di = new DirectoryInfo(debug_config.image_dir);

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
            Directory.CreateDirectory(di.FullName);
        }
    }

    public void ProcessConfig<T>(ref T config, string dir)
    {
        using (StreamReader r = new StreamReader(dir))
        {
            string json = r.ReadToEnd();
            config = JsonConvert.DeserializeObject<T>(json);
        }
    }

    private void ParseCommandLineArguments(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "server":
                    string[] parts = args[++i].Split(':');

                    if (parts.Length == 2)
                    {
                        _instance.command_args.server_info.ip = parts[0];
                        _instance.command_args.server_info.port = Int32.Parse(parts[1]);
                    }
                    break;
                case "debug_conf_dir":
                    _instance.debug_config_dir = args[++i];
                    break;

                case "network_conf_dir":
                    _instance.network_config_dir = args[++i];
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
