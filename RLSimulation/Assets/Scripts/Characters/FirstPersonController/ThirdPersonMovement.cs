using Cinemachine;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;


[RequireComponent(typeof(Rigidbody))]
public class ThirdPersonMovement : MonoBehaviour
{
    private BoxCollider m_collider;

    private float hover_force_equilibrium;

    [HideInInspector]
    public List<Transform> target_transforms = new List<Transform>();

    private float originalDrag;
    private float originalAngularDrag;
    public float waterDragConstant;
    public float waterAngularDragConstant;

    private Rigidbody m_RigidBody;
    [HideInInspector]
    public bool m_Hovering = false;

    public LayerMask groundMask;
    public LayerMask waterMask;
    private List<string> collision_objects = new List<string>();
    private bool m_IsUnderwater;

    public ThirdPersonControlSettings movement_controls = new ThirdPersonControlSettings();

    public Camera firstPersonCam;
    public Camera thirdPersonCam;
    public CinemachineFreeLook cinecamera;

    [HideInInspector] 
    public Camera active_cam, inactive_cam;

    public Tuple<int, int> resolution;

    public Vector3 desiredMove;
    public Vector3 desiredRotation;

    private void CalculateHoverForce()
    {
        float max_volume_displacement = (m_collider.size.x * m_collider.size.y * m_collider.size.z);
        hover_force_equilibrium = (WaterDynamics.water_density * WaterDynamics.g * max_volume_displacement) - (m_RigidBody.mass * WaterDynamics.g); // Fb = pgV - mg
    }

    private void Start()
    {
        m_collider = GetComponentInChildren<BoxCollider>();
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        m_RigidBody = GetComponent<Rigidbody>();
        originalDrag = m_RigidBody.drag;
        originalAngularDrag = m_RigidBody.angularDrag;
        active_cam = thirdPersonCam;
        inactive_cam = firstPersonCam;
        SimulationManager._instance.rover = gameObject;

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.server_config.is_overridden)
        {
            firstPersonCam.fieldOfView = SimulationManager._instance.server.server_config.payload.roverConfig.camConfig.fov;
            resolution = new Tuple<int, int>(SimulationManager._instance.server.server_config.payload.roverConfig.camConfig.resolution[0],
                SimulationManager._instance.server.server_config.payload.roverConfig.camConfig.resolution[1]);
            m_RigidBody.mass += SimulationManager._instance.server.server_config.payload.roverConfig.structureConfig.ballastMass;
        }

        CalculateHoverForce();
    }

    void Update()
    {
        movement_controls.Update(SimulationManager._instance.in_manual_mode);

        ref float fov = ref cinecamera.m_Lens.FieldOfView;
        fov -= movement_controls.mouseWheel * movement_controls.sensitivity;
        fov = Mathf.Clamp(fov, movement_controls.minFov, movement_controls.maxFov);

        if (movement_controls.cameraChange)
        {
            SwitchActiveCamera(active_cam, inactive_cam);
        }

        if (movement_controls.hovering)
        {
            m_Hovering = !m_Hovering;
        }

        if (movement_controls.increase_far_plane && firstPersonCam.farClipPlane < 2000)
        { 
            firstPersonCam.farClipPlane += 100;
            cinecamera.m_Lens.FarClipPlane += 100;
        }

        if (movement_controls.decrease_far_plane && firstPersonCam.farClipPlane > 100)
        {
            firstPersonCam.farClipPlane -= 100;
            cinecamera.m_Lens.FarClipPlane -= 100;
        }
    }

    private void SwitchActiveCamera(Camera active, Camera inactive)
    {
        inactive.depth = 0;
        inactive.rect = new Rect(0, 0, 1, 1);
        active.depth = 1;
        active.rect = new Rect(0.7f, 0, 0.3f, 0.3f);
        active_cam = inactive;
        inactive_cam = active;
    }

    private void FixedUpdate()
    {
        desiredMove = new Vector3();
        desiredRotation = new Vector3();

        m_RigidBody.drag = originalDrag * (m_IsUnderwater ? waterDragConstant : 1);
        m_RigidBody.angularDrag = originalAngularDrag * (m_IsUnderwater ? waterAngularDragConstant : 1);

        if (m_IsUnderwater)
        {
            /* Movement */
            if (movement_controls.movementInputs.magnitude > float.Epsilon)
            {
                if (movement_controls.movementInputs.y != 0)
                {
                    desiredMove += firstPersonCam.transform.up * movement_controls.movementInputs.y * 2;    // Vertical thrusters more power to overcome buoyancy
                }
                if (movement_controls.movementInputs.z != 0)
                {
                    desiredMove += firstPersonCam.transform.forward * movement_controls.movementInputs.z;
                }

                desiredMove *= movement_controls.ThrustPower / 2;
            }

            if (m_Hovering)
            {
                desiredMove.y -= hover_force_equilibrium;
            }

            /* Rotation */
            if (movement_controls.rotationInputs.magnitude > float.Epsilon)
            {
                if (movement_controls.rotationInputs.y != 0)
                {
                    desiredRotation.y += movement_controls.rotationInputs.y;
                }

                desiredRotation *= movement_controls.ThrustPower / 500;
            }

            m_RigidBody.AddForce(desiredMove, ForceMode.Force);
            m_RigidBody.AddRelativeTorque(desiredRotation, ForceMode.Force);
        }
    }

    private void LateUpdate()
    {
        if (!SimulationManager._instance.in_manual_mode && SimulationManager._instance.server.ready_to_send)
        {
            SimulationManager._instance.server.ready_to_send = false;
            SendImageData();
        }
    }

    [Serializable]
    public struct DataToSend<T>
    {
        public string msg_type;
        public T payload;
    }

    [Serializable]
    public struct Telemetary_Data
    {
        public int sequence_num;
        public byte[] jpg_image;
        public Vector3 position;
        public string[] collision_objects;
        public Vector3 fwd;
        public TargetObject[] targets;
    }

    [Serializable]
    public struct TargetObject
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

        TargetObject[] targetPositions = new TargetObject[target_transforms.Count];
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
                sequence_num = SimulationManager._instance.server.sequence_num,
                jpg_image = screenShot.EncodeToJPG(),
                position = transform.position,
                collision_objects = collision_objects.ToArray(),
                fwd = transform.forward,
                targets = targetPositions,
            }
        };

        SimulationManager._instance.server.SendDataAsync(data);
    }

    private void OnCollisionEnter(Collision collision)
    {
        collision_objects.Add(collision.gameObject.tag);
    }

    private void OnCollisionStay(Collision collision)
    {
    }

    private void OnCollisionExit(Collision collision)
    {
        collision_objects.Remove(collision.gameObject.tag);
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
