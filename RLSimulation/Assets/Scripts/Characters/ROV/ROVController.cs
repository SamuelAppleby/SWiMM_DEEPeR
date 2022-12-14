using Cinemachine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using static Server;

[RequireComponent(typeof(Rigidbody))][RequireComponent(typeof(FloaterContainer))]
public class ROVController : MonoBehaviour
{
    [HideInInspector]
    public List<Transform> target_transforms = new List<Transform>();

    public float air_drag;
    public float angular_air_drag;
    public float water_drag;
    public float angular_water_drag;

    [HideInInspector]
    public Rigidbody m_RigidBody;
    [HideInInspector]
    public bool m_depth_hold_mode = true;
    [HideInInspector]
    public bool m_stabilise_mode = true;

    public LayerMask water_mask;
    public LayerMask ground_mask;

    private List<string> collision_objects_list = new List<string>();
    private bool m_IsUnderwater;
    private bool m_is_grounded;

    public ROVControls movement_controls = new ROVControls();

    public Camera firstPersonCam;
    public Camera thirdPersonCam;
    public CinemachineFreeLook cinecamera;

    [HideInInspector] 
    public Camera active_cam, inactive_cam;

    public Tuple<int, int> resolution;

    [HideInInspector]
    public Vector3 desiredMove;

    [HideInInspector]
    public Vector3 desiredRotation;

    public Material underwater_skybox_mat;
    public Material ground_skybox_mat;

    public PostProcessVolume volume;

    private ColorGrading m_colour_grading;

    private Color m_color_grading_filter_start;
    private float max_depth = 1000;
    public GameObject water_collider;
    private float top_of_water;
    private float m_distance_undewater;

    public Vector3 linear_force_to_be_applied;
    public Vector3 last_action_linear_force;
    public Vector3 angular_force_to_be_applied;
    public Vector3 last_action_angular_force;

    private int manual_screenshot_count = 0;

    List<Tuple<int, int>> valid_resolutions = new List<Tuple<int, int>> { new Tuple<int, int>(256, 256), new Tuple<int, int>(512, 512), new Tuple<int, int>(1024, 1024),
    new Tuple<int, int>(2048, 2048)};

    [Serializable]
    public struct Controls_Audio
    {
        public AudioSource audio_motor;
        public AudioSource hover_noise;
    }

    public Controls_Audio audios;

    public void OnJsonControls(JsonMessage param)
    {
        linear_force_to_be_applied = new Vector3(param.payload.jsonControls.swayThrust, 
            param.payload.jsonControls.heaveThrust, param.payload.jsonControls.surgeThrust);
        angular_force_to_be_applied = new Vector3(param.payload.jsonControls.pitchThrust, 
            param.payload.jsonControls.yawThrust, param.payload.jsonControls.rollThrust);

        /* Simulate joystick output with boolean values */
        angular_force_to_be_applied.x = angular_force_to_be_applied.x < 0.5 && angular_force_to_be_applied.x > -0.5 ? 0 : angular_force_to_be_applied.x <= -0.5 ? -1 : 1;    
        angular_force_to_be_applied.z = angular_force_to_be_applied.z < 0.5 && angular_force_to_be_applied.z > -0.5 ? 0 : angular_force_to_be_applied.z <= -0.5 ? -1 : 1;

        if (param.payload.jsonControls.depthHoldMode < 0 && m_depth_hold_mode || param.payload.jsonControls.depthHoldMode >= 0 && !m_depth_hold_mode)
        {
            ToggleDepthHoldMode();
        }

        last_action_linear_force = linear_force_to_be_applied;
        last_action_angular_force = angular_force_to_be_applied;
        StartCoroutine(SendImageData());
    }

    public void ToggleDepthHoldMode()
    {
        audios.hover_noise.Play();
        m_depth_hold_mode = !m_depth_hold_mode;
    }

    private IEnumerator Start()
    {
        top_of_water = water_collider.transform.position.y + (water_collider.GetComponent<BoxCollider>().size.y / 2);
        volume.profile.TryGetSettings(out m_colour_grading);
        m_color_grading_filter_start = m_colour_grading.colorFilter.value;

        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        m_RigidBody = GetComponent<Rigidbody>();
        m_depth_hold_mode = true;
        m_RigidBody.drag = air_drag;
        m_RigidBody.angularDrag = angular_air_drag;
        active_cam = thirdPersonCam;
        inactive_cam = firstPersonCam;

        firstPersonCam.fieldOfView = 100;
        resolution = new Tuple<int, int>(2048,2048);

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.json_server_config.msgType.Length > 0)
        {
            firstPersonCam.fieldOfView = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.fov;
            resolution = new Tuple<int, int>(SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[0],
                SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.resolution[1]);
            m_RigidBody.mass += SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.structureConfig.ballastMass;
            movement_controls.stabilityThreshold = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.stabilityThreshold;
            movement_controls.stabilityForce = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.stabilityForce;
            movement_controls.LinearThrustStrength = Utils.FloatArrayToVector3(ref SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.linearThrustPower);
            movement_controls.AngularThrustStrength = Utils.FloatArrayToVector3(ref SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.angularThrustPower);
        }

        yield return new WaitUntil(() => GetComponent<FloaterContainer>().is_initialized);
        EventMaster._instance.rov_initialised_event.Raise(gameObject);
    }

    public void PlayAudioEffects()
    {
        if (linear_force_to_be_applied.magnitude > 0 || angular_force_to_be_applied.magnitude > 0 || (SimulationManager._instance.server != null && (
            (Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN ||
            Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN_FREEZE) && 
            last_action_linear_force.magnitude > 0 || last_action_angular_force.magnitude > 0)))
        {
            if (!audios.audio_motor.isPlaying)
            {
                audios.audio_motor.Play();
            }
        }
        else
        {
            audios.audio_motor.Stop();
        }
    }

    public void OnMouseWheelScroll(float scroll)
    {
        ref float fov = ref cinecamera.m_Lens.FieldOfView;
        fov -= scroll * movement_controls.sensitivity;
        fov = Mathf.Clamp(fov, movement_controls.minFov, movement_controls.maxFov);
    }

    public void OnChangeCamera()
    {
        inactive_cam.depth = 0;
        inactive_cam.rect = new Rect(0, 0, 1, 1);
        active_cam.depth = 1;
        active_cam.rect = new Rect(0.7f, 0, 0.3f, 0.3f);
        Camera temp = active_cam;
        active_cam = inactive_cam;
        inactive_cam = temp;
        gameObject.GetComponentInChildren<CinemachineFreeLook>().enabled = active_cam == thirdPersonCam;
    }

    public void ChangeFarPlane(float val)
    {
        if ((val > 0 && firstPersonCam.farClipPlane < 2000) || (val < 0 && firstPersonCam.farClipPlane > 100))
        {
            firstPersonCam.farClipPlane += val;
            cinecamera.m_Lens.FarClipPlane += val;
        }
    }

    void Update()
    {
        m_distance_undewater = top_of_water - transform.position.y;
        CheckCameraEffects();
        movement_controls.Update(SimulationManager._instance.in_manual_mode, this);
        PlayAudioEffects();
    }

    private void FixedUpdate()
    {
        desiredMove = new Vector3();
        desiredRotation = new Vector3();
        m_RigidBody.drag = m_IsUnderwater ? water_drag : air_drag;
        m_RigidBody.angularDrag = m_IsUnderwater ? angular_water_drag : angular_air_drag;

        if (SimulationManager._instance.server != null && !SimulationManager._instance.in_manual_mode)
        {
            if(Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN ||
            Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN_FREEZE)
            {
                linear_force_to_be_applied = last_action_linear_force;
                angular_force_to_be_applied = last_action_angular_force;
            }
        } 
            
        /* N.B. Before internal physics engine does oncollisions, provide the check anyway otherwise we skip a frame */
        if (m_IsUnderwater || m_depth_hold_mode)
        {
            /* Movement */
            if (linear_force_to_be_applied.magnitude > float.Epsilon)
            {
                if (linear_force_to_be_applied.x != 0)
                {
                    desiredMove += firstPersonCam.transform.right * linear_force_to_be_applied.x;
                }
                if (linear_force_to_be_applied.y != 0)
                {
                    desiredMove += firstPersonCam.transform.up * linear_force_to_be_applied.y;
                }
                if (linear_force_to_be_applied.z != 0)
                {
                    desiredMove += firstPersonCam.transform.forward * linear_force_to_be_applied.z;
                }
            }

            /* Rotation */
            if (angular_force_to_be_applied.magnitude > float.Epsilon)
            {
                if (angular_force_to_be_applied.x != 0)
                {
                    desiredRotation.x += angular_force_to_be_applied.x;
                }
                if (angular_force_to_be_applied.y != 0)
                {
                    desiredRotation.y += angular_force_to_be_applied.y;
                }
                if (angular_force_to_be_applied.z != 0)
                {
                    desiredRotation.z += angular_force_to_be_applied.z;
                }
            }

            /*
             * 
             * The modes below are for the different settings the rover can be in, don't consider the thrust strength here
             * 
             */

            /* Counteract the forces due to gravity irrelevant of fixed dt */
            if (m_depth_hold_mode)
            {
                m_RigidBody.AddForce(-Physics.gravity, ForceMode.Acceleration);     // ACCELERATION IS A CONSTANT ACCELERATION, TREAT DIFFERENTLY

                //if (m_RigidBody.velocity.y < -movement_controls.stabilityThreshold)
                //{
                //    desiredMove += Vector3.up;
                //}
                //else if (m_RigidBody.velocity.y > movement_controls.stabilityThreshold)
                //{
                //    desiredMove += -Vector3.up;
                //}

                m_RigidBody.AddForce(Vector3.up * -GetComponent<FloaterContainer>().submerged_buoyant_strength, ForceMode.Force);
                //desiredMove += Vector3.up * -GetComponent<FloaterContainer>().submerged_buoyant_strength;
            }

            /* Stabilise roll and pitch, NOT sway */
            if (m_stabilise_mode)
            {
                if(90 - gameObject.transform.rotation.eulerAngles.x < -movement_controls.stabilityThreshold)     // Pitch is too far up so force down
                {
                    desiredRotation += Vector3.right;
                }
                else if (90 - gameObject.transform.rotation.eulerAngles.x > movement_controls.stabilityThreshold)     // Pitch is too far down so force up
                {
                    desiredRotation += -Vector3.right;
                }

                if (90 - gameObject.transform.rotation.eulerAngles.z < -movement_controls.stabilityThreshold)     // Pitch is too far up so force down
                {
                    desiredRotation += Vector3.forward;
                }
                else if (90 - gameObject.transform.rotation.eulerAngles.z > movement_controls.stabilityThreshold)     // Pitch is too far down so force down
                {
                    desiredRotation += -Vector3.forward;
                }
            }

            desiredMove = Vector3.Scale(desiredMove, movement_controls.LinearThrustStrength);
            desiredRotation = Vector3.Scale(desiredRotation, movement_controls.AngularThrustStrength);
            m_RigidBody.AddForce(desiredMove, ForceMode.Force);
            m_RigidBody.AddRelativeTorque(desiredRotation, ForceMode.Force);
        }

        linear_force_to_be_applied = Vector3.zero;
        angular_force_to_be_applied = Vector3.zero;
    }

    private void LateUpdate()
    {
        if (Input.GetKeyUp(KeyCode.V))
        {
            TakeScreenshot(valid_resolutions[manual_screenshot_count], true);
        }
    }

    private Texture2D current_screenshot;

    public void OnAIGroupsComplete()
    {
        StartCoroutine(SendImageData());
    }

    private IEnumerator TakeScreenshot(Tuple<int,int> res, bool save_image)
    {
        yield return new WaitForEndOfFrame();

        RenderTexture rt = new RenderTexture(res.Item1, res.Item2, 24);
        firstPersonCam.targetTexture = rt;
        firstPersonCam.Render();
        RenderTexture.active = rt;
        Texture2D screen_shot = new Texture2D(res.Item1, res.Item2, TextureFormat.RGB24, false);
        screen_shot.ReadPixels(new Rect(0, 0, res.Item1, res.Item2), 0, 0);
        screen_shot.Apply();
        firstPersonCam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        if (save_image)
        {
            File.WriteAllBytes(SimulationManager._instance.debug_config.image_dir + "sent_image" + manual_screenshot_count.ToString() + ".jpg", screen_shot.EncodeToJPG());
            manual_screenshot_count++;
        }

        current_screenshot = screen_shot;
    }

    private IEnumerator SendImageData()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.IsConnectionValid() && !SimulationManager._instance.in_manual_mode)
        {
            yield return StartCoroutine(TakeScreenshot(resolution, SimulationManager._instance.debug_config.save_images));

            TargetObject[] targetPositions = new TargetObject[target_transforms.Count];
            int pos = 0;

            foreach (Transform trans in target_transforms)
            {
                targetPositions[pos] = new TargetObject
                {
                    position = new float[] { trans.position.x, trans.position.y, trans.position.z },
                    fwd = new float[] { trans.forward.x, trans.forward.y, trans.forward.z }
                };
                pos++;
            }

            SimulationManager._instance.server.obsv = new DataToSend
            {
                msg_type = "on_telemetry",
                payload = new Telemetary_Data
                {
                    sequence_num = SimulationManager._instance.server.observations_sent,
                    jpg_image = current_screenshot.EncodeToJPG(),
                    position = new float[] { transform.position.x, transform.position.y, transform.position.z },
                    collision_objects = collision_objects_list.ToArray(),
                    fwd = new float[] { transform.forward.x, transform.forward.y, transform.forward.z },
                    targets = targetPositions
                }
            };
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        collision_objects_list.Add(collision.gameObject.tag);
    }

    private void OnCollisionStay(Collision collision)
    {
    }

    private void OnCollisionExit(Collision collision)
    {
        /* Physics ticks may happen before message is sent, so persist reset on collion */
        //collision_objects_list.Remove(collision.gameObject.tag);
    }

    private void OnTriggerEnter(Collider other)
    {
        m_is_grounded = ((1 << other.gameObject.layer) & ground_mask) != 0;
    }

    private void OnTriggerStay(Collider other)
    {
        m_IsUnderwater = ((1 << other.gameObject.layer) & water_mask) != 0;
    }

    private void OnTriggerExit(Collider other)
    {
        if (m_IsUnderwater && ((1 << other.gameObject.layer) & water_mask) != 0)
        {
            m_IsUnderwater = false;
            m_depth_hold_mode = false;
        }

        m_is_grounded = ((1 << other.gameObject.layer) & ground_mask) != 0;
    }

    public void CheckCameraEffects()
    {
        Collider[] hit_colliders = Physics.OverlapSphere(firstPersonCam.transform.position, 0.1f);

        foreach (Collider col in hit_colliders)
        {
            if (((1 << col.gameObject.layer) & water_mask) != 0)
            {
                //RenderSettings.fog = true;
                RenderSettings.skybox = underwater_skybox_mat;

                if(m_distance_undewater > 0)
                {
                    float depth_ratio_underwater_clamped = 1 - Math.Clamp(m_distance_undewater / max_depth, 0, 1);

                    m_colour_grading.colorFilter.value.r = m_color_grading_filter_start.r * depth_ratio_underwater_clamped;
                    m_colour_grading.colorFilter.value.g = m_color_grading_filter_start.g * depth_ratio_underwater_clamped;
                    m_colour_grading.colorFilter.value.b = m_color_grading_filter_start.b * depth_ratio_underwater_clamped;
                }

                return;
            }
        }

        //RenderSettings.fog = false;
        RenderSettings.skybox = ground_skybox_mat;
    }
}
