using Cinemachine;
using System;
using System.Reflection;
using System.Text;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class ThirdPersonMovement : MonoBehaviour
{
    public Transform[] targetTransforms;
    private Server server;

    private float originalDrag;
    private float originalAngularDrag;
    public float waterDragConstant;
    public float waterAngularDragConstant;
    public float groundDragConstant;
    public float groundAngularDragConstant;

    public Material roverMaterial;
    private int avgFrameRate;

    private Rigidbody m_RigidBody;
    private bool m_Hovering = false;

    public LayerMask groundMask;
    public LayerMask waterMask;
    private bool m_IsGrounded = false;
    private bool m_IsColliding = false;
    public bool m_IsUnderwater = false;

    public ControlSettings movementSettings = new ControlSettings();

    public Camera firstPersonCam;
    public Camera thirdPersonCam;
    public CinemachineFreeLook cinecamera;
    [HideInInspector] public Camera activeCamera;

    public int resWidth = 256;
    public int resHeight = 256;

    private FullScreenMode[] screenmodes;
    private int screenIndex = 0;

    private async void Start()
    {
        m_RigidBody = GetComponent<Rigidbody>();
        originalDrag = m_RigidBody.drag;
        originalAngularDrag = m_RigidBody.angularDrag;
        activeCamera = firstPersonCam;

        screenmodes = new FullScreenMode[] { FullScreenMode.MaximizedWindow, FullScreenMode.FullScreenWindow, FullScreenMode.MaximizedWindow, FullScreenMode.Windowed };
        Screen.fullScreenMode = screenmodes[screenIndex];

        if (SimulationManager.Instance.useServer)
        {
            server = new Server(SimulationManager.Instance.server.URL, SimulationManager.Instance.server.Port, SimulationManager.Instance.server.TickRate);
            await server.Connect();

            if (server.IsTcpGood())
            {
                AwaitAnyServerData();
            }
        }
    }

    private void UpdateFPS()
    {
        float current = (int)(1f / Time.unscaledDeltaTime);
        avgFrameRate = (int)current;
    }

    private void TerminateApplication()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
         Application.Quit();
#endif
    }

    void Update()
    {
        if (SimulationManager.Instance.useServer)
        {
            server.Update(Time.deltaTime);
        }

        movementSettings.Update(SimulationManager.Instance.useServer);

        if (movementSettings.quitting)
        {
            TerminateApplication();
        }

        if (movementSettings.changeWindow)
        {
            screenIndex = screenIndex == screenmodes.Length - 1 ? 0 : screenIndex + 1;
            Screen.fullScreenMode = screenmodes[screenIndex];
        }

        ref float fov = ref cinecamera.m_Lens.FieldOfView;
        fov -= movementSettings.mouseWheel * movementSettings.sensitivity;
        fov = Mathf.Clamp(fov, movementSettings.minFov, movementSettings.maxFov);

        if (movementSettings.cameraChange)
        {
            firstPersonCam.enabled = !firstPersonCam.enabled;
            thirdPersonCam.enabled = !thirdPersonCam.enabled;

            if (!SimulationManager.Instance.useServer)
            {
                activeCamera = thirdPersonCam.enabled ? thirdPersonCam : firstPersonCam;
            }
        }

        if (movementSettings.hovering)
        {
            m_Hovering = !m_Hovering;
            roverMaterial.color = m_Hovering ? Color.green : Color.blue;
        }

        UpdateFPS();
    }

    private void FixedUpdate()
    {
        m_RigidBody.drag = originalDrag * (m_IsGrounded ? groundDragConstant : m_IsUnderwater ? waterDragConstant : 1);
        m_RigidBody.angularDrag = originalAngularDrag * (m_IsGrounded ? groundAngularDragConstant : m_IsUnderwater ? waterAngularDragConstant : 1);

        if (!m_IsUnderwater)
        {
            return;
        }

        if (m_Hovering)
        {
            m_RigidBody.AddForce(new Vector3(0, 9.81f, 0), ForceMode.Acceleration);
        }

        /* Movement */
        if (movementSettings.movementInputs.magnitude > float.Epsilon && (m_RigidBody.velocity.sqrMagnitude < Mathf.Pow(movementSettings.ThrustPower, 2)))
        {
            Vector3 desiredMove = new Vector3();
            if (movementSettings.movementInputs.y != 0 && m_IsUnderwater)
            {
                desiredMove += firstPersonCam.transform.up * movementSettings.movementInputs.y;
            }
            if (movementSettings.movementInputs.z != 0)
            {
                desiredMove += firstPersonCam.transform.forward * movementSettings.movementInputs.z;
            }

            desiredMove *= movementSettings.ThrustPower / 2;

            m_RigidBody.AddForce(desiredMove, ForceMode.Impulse);
        }

        /* Rotation */
        if (movementSettings.rotationInputs.magnitude > float.Epsilon && m_RigidBody.angularVelocity.sqrMagnitude < Mathf.Pow(movementSettings.ThrustPower / 10, 2))
        {
            Vector3 desiredRotation = new Vector3();
            if (movementSettings.rotationInputs.y != 0)
            {
                desiredRotation.y += movementSettings.rotationInputs.y;
            }

            desiredRotation *= movementSettings.ThrustPower / 1000;

            m_RigidBody.AddRelativeTorque(desiredRotation, ForceMode.Impulse);
        }

    }

    private void LateUpdate()
    {
        if (SimulationManager.Instance.useServer)
        {
            if (!server.IsTcpGood())
            {
                return;
            }
            else if (server.GoodToSend())
            {
                SendImageData();
                return;
            }
        }       
    }

    [Serializable]
    struct DataToSend<T>
    {
        public string msg_type;
        public T payload;
    }

    [Serializable]
    struct ImageData
    {
        public byte[] jpgImage;
        public Vector3 currentPosition;
        public Vector3[] targetPositions;
        public bool isColliding;
    }


    [Serializable]
    struct MessageType
    {
        public string msgType;
    }

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

    [Serializable]
    public struct JsonControls
    {
        public float forwardThrust;
        public float verticalThrust;
        public float yRotation;
    }

    public void ProcessServerConfig(ConfigOptions config)
    {
        firstPersonCam.fieldOfView = config.camConfig.fov;
        Fog.SetFogStart(config.envConfig.fogStart);
    }

    public void ReceiveJsonControls(JsonControls controls)
    {
        movementSettings.ReceiveJsonControls(controls);
    }

    private async void AwaitAnyServerData()
    {
        string jsonStr = await server.AwaitAnyData();

        try
        {
            if (jsonStr != null)
            {
                MessageType message = JsonUtility.FromJson<MessageType>(jsonStr);

                try
                {
                    MethodInfo methodName = this.GetType().GetMethod(message.msgType);

                    switch (methodName.Name)
                    {
                        case "ProcessServerConfig":
                            JsonMessage<ConfigOptions> config = JsonUtility.FromJson<JsonMessage<ConfigOptions>>(jsonStr);
                            methodName.Invoke(this, new object[] { config.payload });
                            break;
                        case "ReceiveJsonControls":
                            JsonMessage<JsonControls> controls = JsonUtility.FromJson<JsonMessage<JsonControls>>(jsonStr);
                            methodName.Invoke(this, new object[] { controls.payload });
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

    private async void SendImageData()
    {
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        firstPersonCam.targetTexture = rt;
        firstPersonCam.Render();
        RenderTexture.active = rt;
        Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
        screenShot.Apply();
        firstPersonCam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        Vector3[] targetPositions = new Vector3[targetTransforms.Length];
        for (int i = 0; i < targetPositions.Length; ++i)
        {
            targetPositions[i] = targetTransforms[i].position;
        }

        DataToSend<ImageData> data = new DataToSend<ImageData>
        {
            msg_type = "on_telemetry",
            payload = new ImageData
            {
                jpgImage = screenShot.EncodeToJPG(),
                currentPosition = transform.position,
                targetPositions = targetPositions,
                isColliding = m_IsColliding
            }
        };

        string result = JsonUtility.ToJson(data);

        /* N.B We want to only send 1 request/response at a time, but dont want to block */
        await server.SendDataAsync(Encoding.UTF8.GetBytes(result));
    }

    void OnDrawGizmos()
    {
    }

    private void OnCollisionEnter(Collision collision)
    {
        m_IsColliding = true;
    }

    private void OnCollisionStay(Collision collision)
    {
        m_IsGrounded = ((1 << collision.gameObject.layer) & groundMask) != 0;
    }

    private void OnCollisionExit(Collision collision)
    {
        m_IsGrounded = false;
        m_IsColliding = false;
    }

    private void OnTriggerEnter(Collider other)
    {
    }

    private void OnTriggerStay(Collider other)
    {
        m_IsUnderwater = ((1 << other.gameObject.layer) & waterMask) != 0;
    }

    private void OnTriggerExit(Collider other)
    {
        m_IsUnderwater = ((1 << other.gameObject.layer) & waterMask) == 0;
    }
}
