using Cinemachine;
using System;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;
using static SimulationManager;

[RequireComponent(typeof(Rigidbody))]
public class ThirdPersonMovement : MonoBehaviour
{
    public Transform[] targetTransforms;

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
    private bool m_IsColliding = false;
    public bool m_IsUnderwater = false;

    public ThirdPersonControlSettings movement_controls = new ThirdPersonControlSettings();

    public Camera firstPersonCam;
    public Camera thirdPersonCam;
    public CinemachineFreeLook cinecamera;
    [HideInInspector] public Camera activeCamera;

    public int resWidth = 256;
    public int resHeight = 256;

    private void Start()
    {
        m_RigidBody = GetComponent<Rigidbody>();
        originalDrag = m_RigidBody.drag;
        originalAngularDrag = m_RigidBody.angularDrag;
        activeCamera = thirdPersonCam;
    }

    void Update()
    {
        movement_controls.Update(SimulationManager._instance.useServer);

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
        if (SimulationManager._instance.useServer)
        {
            if (!SimulationManager._instance.server.IsTcpGood())
            {
                return;
            }
            else if (SimulationManager._instance.server.GoodToSend())
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
        public byte[] jpg_image;
        public Vector3 current_position;
        public Vector3[] target_positions;
        public bool is_colliding;
    }

    public void ReceiveJsonControls(JsonControls controls)
    {
        movement_controls.ReceiveJsonControls(controls);
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
                jpg_image = screenShot.EncodeToJPG(),
                current_position = transform.position,
                target_positions = targetPositions,
                is_colliding = m_IsColliding
            }
        };

        string result = JsonUtility.ToJson(data);

        /* N.B We want to only send 1 request/response at a time, but dont want to block */
        await SimulationManager._instance.server.SendDataAsync(Encoding.UTF8.GetBytes(result));
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
