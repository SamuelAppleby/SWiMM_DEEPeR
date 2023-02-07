using Cinemachine;
using System;
using UnityEngine;
using static Server;

public static class MovementControlMap
{
    /* Movement */
    public static string key_surge = "Vertical";
    public static string key_sway = "Horizontal";
    public static KeyCode key_heave_up = KeyCode.Space;
    public static KeyCode key_heave_down = KeyCode.LeftControl;

    /* Rotation */
    public static string key_yaw = "Yaw";
    public static string key_pitch = "Pitch";
    public static KeyCode key_roll_left = KeyCode.Q;
    public static KeyCode key_roll_right = KeyCode.E;

    public static KeyCode key_camera_change = KeyCode.C;

    public static KeyCode key_mode_manual = KeyCode.Alpha1;
    public static KeyCode key_mode_depth_hold = KeyCode.Alpha2;
    public static KeyCode key_mode_stabilize = KeyCode.Alpha3;

    public static KeyCode key_debug_increase_far_plane = KeyCode.Equals;
    public static KeyCode key_debug_decrease_far_plane = KeyCode.Minus;
}

[RequireComponent(typeof(ROVController))]
public class ROVControls : MonoBehaviour
{
    public float stability_force = 0.2f;
    public float stability_threshold = 1f;

    public Vector3 linear_thrust_stength = new Vector3(10, 10, 10);
    public Vector3 angular_thrust_strength = new Vector3(10, 10, 10);

    public AnimationCurve slope_curve_modifier = new AnimationCurve(new Keyframe(-90.0f, 1.0f), new Keyframe(0.0f, 1.0f), new Keyframe(90.0f, 0.0f));

    public float min_fov = 15f;
    public float max_fov = 120f;
    public float sensitivity = 20f;

    [HideInInspector]
    public Vector3 input_linear, last_action_input_linear, input_angular, last_action_input_angular;

    [HideInInspector]
    public Enums.E_Rover_Dive_Mode dive_mode;

    public CinemachineFreeLook cine_cam;

    [Serializable]
    public struct Controls_Audio
    {
        public AudioSource thrusters;
        public AudioSource mode_change;
    }

    public Controls_Audio audios;

    public Camera first_person_cam;
    public Camera third_person_cam;

    [HideInInspector]
    public Camera active_cam, inactive_cam;

    [HideInInspector]
    public Vector3 desired_move;

    [HideInInspector]
    public Vector3 desired_rotation;

    private Rigidbody applicable_body;

    [HideInInspector]
    public bool object_underwater = false;

    public void Start()
    {
        dive_mode = Enums.E_Rover_Dive_Mode.DEPTH_HOLD;
        applicable_body = GetComponent<Rigidbody>();
        active_cam = first_person_cam.depth == 0 ? first_person_cam : third_person_cam;
        inactive_cam = active_cam == first_person_cam ? third_person_cam : first_person_cam;

        gameObject.GetComponentInChildren<CinemachineFreeLook>().enabled = active_cam == third_person_cam;

        if (SimulationManager._instance.server != null && SimulationManager._instance.server.json_server_config.msgType.Length > 0)
        {
            first_person_cam.fieldOfView = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.camConfig.fov;
            stability_threshold = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.stabilityThreshold;
            stability_force = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.stabilityForce;
            linear_thrust_stength = Utils.FloatArrayToVector3(SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.linearThrustPower);
            angular_thrust_strength = Utils.FloatArrayToVector3(SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.motorConfig.angularThrustPower);
        }
    }

    public void Update()
    {
        object_underwater = GetComponent<ROVController>().is_underwater;

        /* Get manual input when not using tcp */
        if (SimulationManager._instance.in_manual_mode)
        {
            input_linear = new Vector3(Input.GetAxisRaw(MovementControlMap.key_sway),
                Input.GetKey(MovementControlMap.key_heave_up) ? 1 : Input.GetKey(MovementControlMap.key_heave_down) ? -1 : 0, Input.GetAxisRaw(MovementControlMap.key_surge));

            input_angular = new Vector3(Input.GetAxisRaw(MovementControlMap.key_pitch),
                Input.GetAxisRaw(MovementControlMap.key_yaw), Input.GetKey(MovementControlMap.key_roll_left) ? 1 : Input.GetKey(MovementControlMap.key_roll_right) ? -1 : 0);

            if (Input.GetKeyDown(MovementControlMap.key_mode_manual))
            {
                ToggleControlMode(Enums.E_Rover_Dive_Mode.MANUAL);
            }

            if (Input.GetKeyDown(MovementControlMap.key_mode_depth_hold))
            {
                ToggleControlMode(Enums.E_Rover_Dive_Mode.DEPTH_HOLD);
            }

            if (Input.GetKeyDown(MovementControlMap.key_mode_stabilize))
            {
                ToggleControlMode(Enums.E_Rover_Dive_Mode.STABILIZE);
            }
        }

        if (Input.GetAxis("Mouse ScrollWheel") != 0)
        {
            OnMouseWheelScroll(Input.GetAxis("Mouse ScrollWheel"));
        }

        if(Input.GetKeyDown(MovementControlMap.key_camera_change))
        {
            OnChangeCamera();
        }

        if (Input.GetKeyDown(MovementControlMap.key_debug_increase_far_plane))
        {
            ChangeFarPlane(100);
        }

        if (Input.GetKeyDown(MovementControlMap.key_debug_decrease_far_plane))
        {
            ChangeFarPlane(-100);
        }

        PlayAudioEffects();
    }

    private void FixedUpdate()
    {
        desired_move = new Vector3();
        desired_rotation = new Vector3();

        if (SimulationManager._instance.server != null && !SimulationManager._instance.in_manual_mode)
        {
            if (Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN ||
            Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN_FREEZE)
            {
                input_linear = last_action_input_linear;
                input_angular = last_action_input_angular;
            }
        }

        /* N.B. Before internal physics engine does oncollisions, provide the check anyway otherwise we skip a frame */
        if (object_underwater)
        {
            /* Movement */
            if (input_linear.magnitude > float.Epsilon)
            {
                if (input_linear.x != 0)
                {
                    desired_move += first_person_cam.transform.right * input_linear.x;
                }
                if (input_linear.y != 0)
                {
                    desired_move += first_person_cam.transform.up * input_linear.y;
                }
                if (input_linear.z != 0)
                {
                    desired_move += first_person_cam.transform.forward * input_linear.z;
                }
            }

            /* Rotation */
            if (input_angular.magnitude > float.Epsilon)
            {
                if (input_angular.x != 0)
                {
                    desired_rotation.x += input_angular.x;
                }
                if (input_angular.y != 0)
                {
                    desired_rotation.y += input_angular.y;
                }
                if (input_angular.z != 0)
                {
                    desired_rotation.z += input_angular.z;
                }
            }

            desired_move = Vector3.Scale(desired_move, linear_thrust_stength);
            desired_rotation = Vector3.Scale(desired_rotation, angular_thrust_strength);

            /*
             * 
             * The modes below are for the different settings the rover can be in, don't consider the thrust strength here
             * 
             */

            /* Stabilise roll and pitch, NOT sway */
            if (dive_mode != Enums.E_Rover_Dive_Mode.MANUAL)
            {
                Vector3 interp_from = Vector3.zero;

                for (int i = 0; i < 3; ++i)
                {
                    /* Don't stabilize x, not available on default BLUEROV 2 configuration, or y as pitch is retrieved action instead */
                    if (/*i == 0 ||*/ i == 1)
                    {
                        continue;
                    }

                    interp_from[i] = gameObject.transform.rotation.eulerAngles[i] > 270 && 360 - gameObject.transform.rotation.eulerAngles[i] > stability_threshold ? (360 - gameObject.transform.rotation.eulerAngles[i]) :
                    gameObject.transform.rotation.eulerAngles[i] < 90 && gameObject.transform.rotation.eulerAngles[i] > stability_threshold ? -gameObject.transform.rotation.eulerAngles[i] : 0;
                }

                desired_rotation += Vector3.Lerp(interp_from, Vector3.zero, 0.01f) * stability_force;

                /* Counteract the forces due to gravity irrelevant of fixed dt */
                if (dive_mode == Enums.E_Rover_Dive_Mode.DEPTH_HOLD)
                {
                    applicable_body.AddForce(-Physics.gravity, ForceMode.Acceleration);     // ACCELERATION IS A CONSTANT ACCELERATION, TREAT DIFFERENTLY

                    foreach (Floater floater in GetComponentsInChildren<Floater>())
                    {
                        desired_move += Vector3.up * -floater.buoyant_strength;
                    }
                }
            }

            applicable_body.AddForce(desired_move, ForceMode.Force);
            applicable_body.AddRelativeTorque(desired_rotation, ForceMode.Force);
        }

        input_linear = Vector3.zero;
        input_angular = Vector3.zero;
    }

    public void PlayAudioEffects()
    {
        if (input_linear.magnitude > 0 || input_angular.magnitude > 0 || (SimulationManager._instance.server != null && (
            (Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN ||
            Enums.action_inference_mapping[SimulationManager._instance.server.json_server_config.payload.serverConfig.envConfig.actionInference] == Enums.E_Action_Inference.MAINTAIN_FREEZE) &&
            last_action_input_linear.magnitude > 0 || last_action_input_angular.magnitude > 0)))
        {
            if (!audios.thrusters.isPlaying)
            {
                audios.thrusters.Play();
            }
        }
        else
        {
            audios.thrusters.Stop();
        }
    }

    public void OnMouseWheelScroll(float scroll)
    {
        ref float fov = ref cine_cam.m_Lens.FieldOfView;
        fov -= scroll * sensitivity;
        fov = Mathf.Clamp(fov, min_fov, max_fov);
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
        gameObject.GetComponentInChildren<CinemachineFreeLook>().enabled = active_cam == third_person_cam;
    }

    public void ChangeFarPlane(float val)
    {
        if ((val > 0 && first_person_cam.farClipPlane < 2000) || (val < 0 && first_person_cam.farClipPlane > 100))
        {
            first_person_cam.farClipPlane += val;
            cine_cam.m_Lens.FarClipPlane += val;
        }
    }

    public void ToggleControlMode(Enums.E_Rover_Dive_Mode to)
    {
        audios.mode_change.Play();
        dive_mode = to;
    }

    public void OnActionReceived(JsonMessage param)
    {
        input_linear = new Vector3(param.payload.jsonControls.swayThrust,
            param.payload.jsonControls.heaveThrust, param.payload.jsonControls.surgeThrust);
        input_angular = new Vector3(param.payload.jsonControls.pitchThrust,
            param.payload.jsonControls.yawThrust, param.payload.jsonControls.rollThrust);

        /* Simulate joystick output with boolean values */
        input_angular.x = input_angular.x < 0.5 && input_angular.x > -0.5 ? 0 : input_angular.x <= -0.5 ? -1 : 1;
        input_angular.z = input_angular.z < 0.5 && input_angular.z > -0.5 ? 0 : input_angular.z <= -0.5 ? -1 : 1;


        /* DEPTH_HOLD takes precedence first, then STABILIZE, then MANUAL */
        if (param.payload.jsonControls.depthHoldMode > 0)
        {
            if (dive_mode != Enums.E_Rover_Dive_Mode.DEPTH_HOLD)
            {
                ToggleControlMode(Enums.E_Rover_Dive_Mode.DEPTH_HOLD);
            }
        }

        else if (param.payload.jsonControls.stabilizeMode > 0)
        {
            if (dive_mode != Enums.E_Rover_Dive_Mode.STABILIZE)
            {
                ToggleControlMode(Enums.E_Rover_Dive_Mode.STABILIZE);
            }
        }

        else if (param.payload.jsonControls.manualMode > 0)
        {
            if (dive_mode != Enums.E_Rover_Dive_Mode.MANUAL)
            {
                ToggleControlMode(Enums.E_Rover_Dive_Mode.MANUAL);
            }
        }

        last_action_input_linear = input_linear;
        last_action_input_angular = input_angular;
    }
}
