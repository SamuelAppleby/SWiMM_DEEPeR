using Cinemachine;
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
using static Enums;
using static Server;
using AsyncOperation = UnityEngine.AsyncOperation;
using Random = UnityEngine.Random;

public class SimulationManager : Singleton<SimulationManager>
{
    public int num_images;

    public EventMaster event_master;

    public E_Game_State game_state;

    [HideInInspector]
    public Enums.E_SceneIndices current_scene_index;
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

    [Serializable]
    public struct DebugConfig
    {
        public string image_dir;
        public string packets_sent_dir;
        public string packets_received_dir;
    }

    /* Local configs for reading */
    [HideInInspector]
    public DebugConfig debug_config;

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct NetworkConfig
    {
        public string protocol;
        public string host;
        public int port;
        public Buffers buffers;
    }

    [HideInInspector]
    public NetworkConfig network_config;

    public void OnObservationSent()
    {
        _instance.server.obsv_num++;

        if (action_inference_mapping[_instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == E_Action_Inference.FREEZE ||
            action_inference_mapping[_instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == E_Action_Inference.MAINTAIN_FREEZE)
        {
            Time.timeScale = 0;
        }
    }

    public void OnActionReceived(JsonMessage param)
    {
        _instance.server.total_steps++;

        if (Time.timeScale == 0)
        {
            Time.timeScale = 1;
        }
    }

    public async void OnServerConnectionResponse(Exception e)
    {
        if(e == null)
        {
            await Task.Run(() => server.ContinueReadWrite());
        }
        // TODO Clean up server caches
    }

    public void OnServerConfigReceived(JsonMessage param)
    {
        _instance.server.json_server_config = param;

        if (_instance.server != null && _instance.server.IsConnectionValid())
        {
            _instance.server.json_str_obsv = JsonConvert.SerializeObject(new DataToSend
            {
                msg_type = "server_config_received"
            });
        }
    }

    public void EpisodeReset(bool in_manual)
    {
        if(_instance.server != null)
        {
            switch (current_scene_index)
            {
                case Enums.E_SceneIndices.MAIN_MENU:
                    _instance.processing_obj.SetActive(false);
                    break;
                case Enums.E_SceneIndices.SIMULATION:
                    if (!in_manual)
                    {
                        Time.timeScale = 0;
                    }

                    break;
            }
        }

        _instance.MoveToScene(Enums.E_SceneIndices.SIMULATION, in_manual);
    }

    public void ExitCurrentScene()
    {
        _instance.MoveToScene(Enums.E_SceneIndices.SIMULATION);       // In this case will unload and reload as intended
    }

    public void EndSimulation()
    {
        _instance.MoveToScene(Enums.E_SceneIndices.EXIT);       // As above
    }

    public void OnROVInitialised(GameObject rov)
    {
        rover = rov;
    }

    public void ConnectToServer(string ip, int port)
    {
        _instance.server = new Server();
        Tuple<Server, Exception> ret = _instance.server.Connect(network_config, ip, port);
        _instance.server = ret.Item1;
        event_master.server_connection_attempt_event.Raise(ret.Item2);
    }

    public void OnServerDisconnected()
    {
        Debug.Log("No server response");
        SceneManager.LoadScene((int)E_SceneIndices.MAIN_MENU);
    }

    private void Start()
    {
        _instance.game_state = E_Game_State.REGULAR;

        _instance.ParseCommandLineArguments(Environment.GetCommandLineArgs());

        _instance.InvokeRepeating("UpdateFPS", 0f, 1);

#if UNITY_EDITOR
        _instance.debug_config_dir = "../Configs/json/debug_config.json";
        _instance.network_config_dir = "../Configs/json/network_config.json";
#else
        _instance.debug_config_dir = "../../../Configs/json/debug_config.json";
        _instance.network_config_dir = "../../../Configs/json/network_config.json";
#endif        

        _instance.debug_config = _instance.ProcessConfig<DebugConfig>(_instance.debug_config_dir);
        _instance.network_config = _instance.ProcessConfig<NetworkConfig>(_instance.network_config_dir);

#if !UNITY_EDITOR
        _instance.debug_config.image_dir = "../../" + _instance.debug_config.image_dir;
        _instance.debug_config.packets_received_dir = "../../" + _instance.debug_config.packets_received_dir;
        _instance.debug_config.packets_sent_dir = "../../" + _instance.debug_config.packets_sent_dir;
#endif

        Utils.CleanAndCreateDirectories(new Dictionary<string, bool>()
                {
                    { _instance.debug_config.image_dir, true },
                    { _instance.debug_config.packets_sent_dir, true },
                    { _instance.debug_config.packets_received_dir, true },
                });

        _instance.screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];

        _instance.server = null;
        _instance.in_manual_mode = true;

        _instance.MoveToScene(Enums.E_SceneIndices.MAIN_MENU);
    }

    protected override void Awake()
    {
        base.Awake();
        event_master = GetComponent<EventMaster>();
    }

    public void MoveToScene(E_SceneIndices to, bool in_manual = false)
    {
        if (to == E_SceneIndices.EXIT)
        {
            Utils.QuitApplication();
            return;
        }

        _instance.background_image.sprite = background_images[Random.Range(0, background_images.Length)];
        _instance.loading_screen.gameObject.SetActive(true);

        _instance.in_manual_mode = in_manual;

        /* We don't want to unload the persistent scene */
        if (_instance.current_scene_index != E_SceneIndices.PERSISTENT_SCENE)
        {
            SceneManager.UnloadSceneAsync((int)_instance.current_scene_index).completed += handle =>
            {
                LoadScene(to, handle);
            };
        }
        else
        {
            LoadScene(to, null);
        }

    }

    private void LoadScene(E_SceneIndices to, AsyncOperation op)
    {
        StartCoroutine(GenerateTips());
        StartCoroutine(GetSceneLoadProgress());

        SceneManager.LoadSceneAsync((int)to, LoadSceneMode.Additive).completed += handle =>
        {
            event_master.scene_changed_event.Raise(to);
        };
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

        while (FishSpawner.current != null)
        {
            _instance.total_spawn_progress = Mathf.Round(FishSpawner.current.current_progress * 100f);
            switch (FishSpawner.current.current_stage)
            {
                case E_InitialisationStage.INITIALISING_NPCS:
                    _instance.text_field.text = string.Format("Initialising NPCs {0}%", _instance.total_spawn_progress);
                    break;
                case E_InitialisationStage.SPAWNING_NPCS:
                    _instance.text_field.text = string.Format("Spawning NPCs {0}%", _instance.total_spawn_progress);
                    break;
            }
        }

        _instance.progress_bar.value = Mathf.RoundToInt(Mathf.Round((_instance.total_scene_progress + _instance.total_spawn_progress) / 2f));
        _instance.loading_screen.gameObject.SetActive(false);

        yield break;
    }

    public void OnSceneChanged(E_SceneIndices to)
    {
        switch (to)
        {
            case E_SceneIndices.SIMULATION:
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
                break;
            default:
                break;
        }

        _instance.scene_loading = null;
        _instance.current_scene_index = to;
    }

    public void IndexWindow()
    {
        _instance.screenIndex = _instance.screenIndex == screenmodes.Length - 1 ? 0 : screenIndex + 1;
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];
    }

    public void IndexCursor()
    {
        Cursor.lockState = Cursor.lockState == CursorLockMode.Locked ? CursorLockMode.None : CursorLockMode.Locked;
        _instance.rover.GetComponentInChildren<CinemachineFreeLook>().enabled = Cursor.lockState == CursorLockMode.Locked;
        Cursor.visible = !Cursor.visible;
    }

    public void ResetNPCs()
    {
        FishSpawner spawn = FindObjectOfType<FishSpawner>();

        if(spawn != null)
        {
            foreach (Transform group_trans in spawn.m_group_parent.GetComponentInChildren<Transform>().transform)
            {
                Destroy(group_trans.gameObject);
            }
        }
    }

    public void OnSetPosition(JsonMessage msg)
    {
        foreach (GameObjectPosition pos in msg.payload.objectPositions)
        {
            switch (pos.object_name)
            {
                case "rover":
                    rover.transform.position = Utils.FloatArrayToVector3(pos.position);
                    rover.transform.rotation = Utils.FloatArrayToQuaternion(pos.rotation);
                    break;
                default:
                    break;
            }
        }
    }

    void Update()
    {
        globalControls.Update(_instance.in_manual_mode);
    }

    public T ProcessConfig<T>(string dir)
    {
        using (StreamReader r = new StreamReader(dir))
        {
            string json = r.ReadToEnd();
            return JsonConvert.DeserializeObject<T>(json);
        }
    }

    private void ParseCommandLineArguments(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "training":
                    _instance.game_state = E_Game_State.AUTOMATION_TRAINING;
                    break;
                case "sample_gen":
                    _instance.game_state = E_Game_State.IMAGE_SAMPLING;
                    break;
                case "vae_gen":
                    _instance.game_state = E_Game_State.VAE_GEN;
                    break;
                case "num_images":
                    _instance.num_images = int.Parse(args[++i]);
                    break;
            }
        }

        _instance.game_state = E_Game_State.VAE_GEN;
        _instance.num_images = 100;

        switch (_instance.game_state)
        {
            case E_Game_State.AUTOMATION_TRAINING:
                _instance.GetComponent<AutomationTraining>().enabled = true;
                break;
            case E_Game_State.IMAGE_SAMPLING:
            case E_Game_State.VAE_GEN:
                _instance.GetComponent<AutomationImageSampling>().enabled = true;
                break;
            default:
                break;
        }
    }

    private void UpdateFPS()
    {
        float current = (int)(1f / Time.unscaledDeltaTime);
        _instance.avgFrameRate = (int)current;
    }
}