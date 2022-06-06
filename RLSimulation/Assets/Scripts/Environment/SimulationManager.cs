using System;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SimulationManager : Singleton<SimulationManager>
{
    public Server server;

    public GameObject rover;

    private int avgFrameRate;

    private FullScreenMode[] screenmodes;
    private int screenIndex = 0;

    public GlobalControlSettings globalControls = new GlobalControlSettings();

    [Serializable]
    struct JsonMessage<T>
    {
        public T payload;
    }

    [Serializable]
    public struct ConfigOptions
    {
        public CameraConfig camConfig;
        public EnvironmentConfig envConfig;
    }

    [Serializable]
    public struct CameraConfig
    {
        public int fov;
    }

    [Serializable]
    public struct EnvironmentConfig
    {
        public float fogStart;
    }

    public bool useServer = false;

    async void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        _instance.screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = _instance.screenmodes[screenIndex];

        ParseCommandLineArguments(System.Environment.GetCommandLineArgs());

        if (_instance.useServer)
        {
            await _instance.server.Connect();

            if (_instance.server.IsTcpGood())
            {
                AwaitAnyServerData();
            }
        }
    }

    void Update()
    {
        _instance.globalControls.Update(_instance.useServer);

        if (_instance.globalControls.quitting)
        {
            TerminateApplication();
        }

        if (_instance.globalControls.reload_scene)
        {
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            _instance.globalControls.reload_scene = false;
        }

        if (_instance.globalControls.changeWindow)
        {
            _instance.screenIndex = _instance.screenIndex == screenmodes.Length - 1 ? 0 : screenIndex + 1;
            Screen.fullScreenMode = _instance.screenmodes[screenIndex];
        }

        if (_instance.useServer)
        {
            _instance.server.Update(Time.deltaTime);
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

    public void ProcessServerConfig(ConfigOptions config)
    {
        _instance.rover.GetComponent<ThirdPersonMovement>().firstPersonCam.fieldOfView = config.camConfig.fov;
        Fog.SetFogStart(config.envConfig.fogStart);
    }

    [Serializable]
    struct MessageType
    {
        public string msgType;
    }

    [Serializable]
    public struct JsonControls
    {
        public float forwardThrust;
        public float verticalThrust;
        public float yRotation;
    }

    [Serializable]
    public struct GlobalCommand
    {
        public bool reset_episode;
    }

    public void ReceiveGlobalCommand(GlobalCommand command)
    {
        _instance.globalControls.OverrideGlobalControls(command);
    }

    private async void AwaitAnyServerData()
    {
        string jsonStr = await _instance.server.AwaitAnyData();

        try
        {
            if (jsonStr != null)
            {
                MessageType message = JsonUtility.FromJson<MessageType>(jsonStr);
                Debug.Log(message);
                try
                {
                    switch (message.msgType)
                    {
                        case "process_server_config":
                            JsonMessage<ConfigOptions> config = JsonUtility.FromJson<JsonMessage<ConfigOptions>>(jsonStr);
                            _instance.ProcessServerConfig(config.payload);
                            break;
                        case "receive_json_controls":
                            JsonMessage<JsonControls> controls = JsonUtility.FromJson<JsonMessage<JsonControls>>(jsonStr);
                            _instance.rover.GetComponent<ThirdPersonMovement>().ReceiveJsonControls(controls.payload);
                            break;
                        case "global_message":
                            JsonMessage<GlobalCommand> reset = JsonUtility.FromJson<JsonMessage<GlobalCommand>>(jsonStr);
                            _instance.ReceiveGlobalCommand(reset.payload);
                            break;
                        default:
                            break;
                    }
                }

                catch (Exception e)
                {
                    Debug.LogException(e);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }

        AwaitAnyServerData();
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
                            _instance.server = new Server(parts[0], Int32.Parse(parts[1]), 1);
                            _instance.useServer = true;
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
