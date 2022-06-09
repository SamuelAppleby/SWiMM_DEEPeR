using Cinemachine;
using System;
using System.Collections.Generic;
using UnityEngine;
using static Server;

[RequireComponent(typeof(Rigidbody))]
public class ThirdPersonMovement : MonoBehaviour
{
    public Transform[] target_transforms;

    private float originalDrag;
    private float originalAngularDrag;
    public float waterDragConstant;
    public float waterAngularDragConstant;
    public float groundDragConstant;
    public float groundAngularDragConstant;

    public Material roverMaterial;

    private Rigidbody m_RigidBody;
    private bool m_Hovering = false;

    public LayerMask groundMask;
    public LayerMask waterMask;
    private bool m_IsGrounded = false;
    private List<string> collision_objects = new List<string>();
    [HideInInspector]
    public bool m_IsUnderwater;

    public ThirdPersonControlSettings movement_controls = new ThirdPersonControlSettings();

    public Camera firstPersonCam;
    public Camera thirdPersonCam;
    public CinemachineFreeLook cinecamera;

    [HideInInspector] 
    public Camera activeCamera;

    public Tuple<int,int> resolution = new Tuple<int, int>(256, 256);

    private void Start()
    {
        m_RigidBody = GetComponent<Rigidbody>();
        originalDrag = m_RigidBody.drag;
        originalAngularDrag = m_RigidBody.angularDrag;
        activeCamera = firstPersonCam;
        SimulationManager._instance.rover = gameObject;

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.server_config.is_overridden)
        {
            firstPersonCam.fieldOfView = SimulationManager._instance.server.server_config.payload.camConfig.fov;
            RenderSettings.fogStartDistance = SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogStart;
            RenderSettings.fogEndDistance = SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogEnd;
            RenderSettings.fog = SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogOn;
        }
    }

    void Update()
    {
        movement_controls.Update(SimulationManager._instance.server != null &&  SimulationManager._instance.server.IsTcpGood());

        ref float fov = ref cinecamera.m_Lens.FieldOfView;
        fov -= movement_controls.mouseWheel * movement_controls.sensitivity;
        fov = Mathf.Clamp(fov, movement_controls.minFov, movement_controls.maxFov);

        if (movement_controls.cameraChange)
        {
            if(activeCamera == firstPersonCam)
            {
                SwitchActiveCamera(ref firstPersonCam, ref thirdPersonCam);
            }
            else
            {
                SwitchActiveCamera(ref thirdPersonCam, ref firstPersonCam);
            }
        }

        if (movement_controls.hovering)
        {
            m_Hovering = !m_Hovering;
            roverMaterial.color = m_Hovering ? Color.green : Color.blue;
        }
    }

    private void SwitchActiveCamera(ref Camera active, ref Camera inactive)
    {
        inactive.depth = 0;
        inactive.rect = new Rect(0, 0, 1, 1);
        active.depth = 1;
        active.rect = new Rect(0.7f, 0, 0.3f, 0.3f);
        activeCamera = inactive;
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
        if (movement_controls.movementInputs.magnitude > float.Epsilon && (m_RigidBody.velocity.sqrMagnitude < Mathf.Pow(movement_controls.ThrustPower, 2)))
        {
            Vector3 desiredMove = new Vector3();
            if (movement_controls.movementInputs.y != 0 && m_IsUnderwater)
            {
                desiredMove += firstPersonCam.transform.up * movement_controls.movementInputs.y;
            }
            if (movement_controls.movementInputs.z != 0)
            {
                desiredMove += firstPersonCam.transform.forward * movement_controls.movementInputs.z;
            }

            desiredMove *= movement_controls.ThrustPower / 2;

            m_RigidBody.AddForce(desiredMove, ForceMode.Impulse);
        }

        /* Rotation */
        if (movement_controls.rotationInputs.magnitude > float.Epsilon && m_RigidBody.angularVelocity.sqrMagnitude < Mathf.Pow(movement_controls.ThrustPower / 10, 2))
        {
            Vector3 desiredRotation = new Vector3();
            if (movement_controls.rotationInputs.y != 0)
            {
                desiredRotation.y += movement_controls.rotationInputs.y;
            }

            desiredRotation *= movement_controls.ThrustPower / 1000;

            m_RigidBody.AddRelativeTorque(desiredRotation, ForceMode.Impulse);
        }

    }

    private void LateUpdate()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.IsTcpGood() && SimulationManager._instance.server.server_config.is_overridden && SimulationManager._instance.server.ready_to_send)
        {
            SendImageData();
            SimulationManager._instance.server.ready_to_send = false;
        }
    }

    [Serializable]
    struct DataToSend<T>
    {
        public string msg_type;
        public T payload;
    }

    [Serializable]
    struct Telemetary_Data
    {
        public byte[] jpg_image;
        public Vector3 position;
        public string[] collision_objects;
        public Vector3 fwd;
        public TargetObject[] targets;
    }

    [Serializable]
    struct TargetObject
    {
        public Vector3 position;
        public Vector3 fwd;
    }

    private void SendImageData()
    {
        RenderTexture rt = new RenderTexture(resolution.Item1, resolution.Item2, 24);
        firstPersonCam.targetTexture = rt;
        firstPersonCam.Render();
        RenderTexture.active = rt;
        Texture2D screenShot = new Texture2D(resolution.Item1, resolution.Item2, TextureFormat.RGB24, false);
        screenShot.ReadPixels(new Rect(0, 0, resolution.Item1, resolution.Item2), 0, 0);
        screenShot.Apply();
        firstPersonCam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        TargetObject[] targetPositions = new TargetObject[target_transforms.Length];
        int pos = 0;

        foreach(Transform trans in target_transforms)
        {
            targetPositions[pos] = new TargetObject
            {
                position = trans.position,
                fwd = trans.forward
            };
            pos++;
        }

        DataToSend<Telemetary_Data> data = new DataToSend<Telemetary_Data>
        {
            msg_type = "on_telemetry",
            payload = new Telemetary_Data
            {
                jpg_image = screenShot.EncodeToJPG(),
                position = transform.position,
                collision_objects = collision_objects.ToArray(),
                fwd = transform.forward,
                targets = targetPositions,
            }
        };

        string result = JsonUtility.ToJson(data);
        SimulationManager._instance.server.SendImageData(result);
    }

    private void OnCollisionEnter(Collision collision)
    {
        collision_objects.Add(LayerMask.LayerToName(collision.gameObject.layer));
    }

    private void OnCollisionStay(Collision collision)
    {
        m_IsGrounded = ((1 << collision.gameObject.layer) & groundMask) != 0;
    }

    private void OnCollisionExit(Collision collision)
    {
        collision_objects.Remove(collision.gameObject.tag);
        m_IsGrounded = false;
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
