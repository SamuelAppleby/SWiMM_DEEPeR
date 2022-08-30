using System;
using UnityEngine;
using static Server;

public static class MovementControlMap
{
    /* Movement */
    public static string ForwardKey = "Vertical";
    public static string StrafeKey = "Horizontal";
    public static KeyCode RiseKey = KeyCode.Space;
    public static KeyCode FallKey = KeyCode.LeftControl;

    /* Rotation */
    public static string YawKey = "Yaw";
    public static string PitchKey = "Pitch";
    public static KeyCode RollPositive = KeyCode.Q;
    public static KeyCode RollNegative = KeyCode.E;

    public static KeyCode CameraChange = KeyCode.C;

    public static KeyCode HoverKey = KeyCode.H;
    public static KeyCode IncreaseFarPlane = KeyCode.Equals;
    public static KeyCode DecreaseFarPlane = KeyCode.Minus;
}

[Serializable]
public struct Controls_Audio
{
    public AudioSource audio_motor;
    public AudioSource hover_noise;
}

[Serializable]
public class ThirdPersonControlSettings
{
    public Controls_Audio audios;
    public float ThrustPower = 8.0f;
    public AnimationCurve SlopeCurveModifier = new AnimationCurve(new Keyframe(-90.0f, 1.0f), new Keyframe(0.0f, 1.0f), new Keyframe(90.0f, 0.0f));

    public float minFov = 15f;
    public float maxFov = 120f;
    public float sensitivity = 20f;

    [HideInInspector] public Vector3 movementInputs;
    [HideInInspector] public Vector3 rotationInputs;
    [HideInInspector] public bool hover_toggle = false;
    [HideInInspector] public float mouseWheel = 0.0f;
    [HideInInspector] public bool cameraChange = false;
    [HideInInspector] public bool increase_far_plane = false;
    [HideInInspector] public bool decrease_far_plane = false;

    public void Update(bool in_manual_mode)
    {
        /* Get manual input when not using tcp */
        if (in_manual_mode)
        {
            movementInputs.x = Input.GetAxisRaw(MovementControlMap.StrafeKey);
            movementInputs.y = Input.GetKey(MovementControlMap.RiseKey) ? 1 : Input.GetKey(MovementControlMap.FallKey) ? -1 : 0;
            movementInputs.z = Input.GetAxisRaw(MovementControlMap.ForwardKey);
            rotationInputs.x = Input.GetAxisRaw(MovementControlMap.PitchKey);
            rotationInputs.y = Input.GetAxisRaw(MovementControlMap.YawKey);
            rotationInputs.z = Input.GetKey(MovementControlMap.RollPositive) ? 1 : Input.GetKey(MovementControlMap.RollNegative) ? -1 : 0;

            hover_toggle = Input.GetKeyDown(MovementControlMap.HoverKey);
        }
        else if (SimulationManager._instance.server.rover_controls.is_overridden)
        {
            // now it depends on what is passed from learningconfig
            movementInputs.x = Input.GetAxisRaw(MovementControlMap.StrafeKey);
            movementInputs.y = SimulationManager._instance.server.rover_controls.payload.verticalThrust;
            movementInputs.z = SimulationManager._instance.server.rover_controls.payload.forwardThrust;
            rotationInputs.x = SimulationManager._instance.server.rover_controls.payload.pitchThrust;
            rotationInputs.x = rotationInputs.x < 0.5 && rotationInputs.x > -0.5 ? 0 : rotationInputs.x <= -0.5 ? -1 : 1;     // Uses Dpad, no intermediate values
            rotationInputs.y = SimulationManager._instance.server.rover_controls.payload.yawThrust;
            rotationInputs.z = SimulationManager._instance.server.rover_controls.payload.rollThrust;
            rotationInputs.z = rotationInputs.z < 0.5 && rotationInputs.z > -0.5 ? 0 : rotationInputs.z <= -0.5 ? -1 : 1;     // Uses Dpad, no intermediate values

            if (SimulationManager._instance.server.rover_controls.payload.depthHoldMode < 0.5 && 
                SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().m_depth_hold_mode ||
                SimulationManager._instance.server.rover_controls.payload.depthHoldMode >= 0.5 &&
                !SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().m_depth_hold_mode)
            {
                hover_toggle = true;
            }

            else
            {
                hover_toggle = false;
            }

            // Sam.A Talk to Kirsten, do we want previous controls for good?
            //SimulationManager._instance.server.rover_controls.Reset();
        }

        /* Audio */
        if (movementInputs.magnitude > 0 || rotationInputs.magnitude > 0)
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
        if (hover_toggle)
        {
            audios.hover_noise.Play();
        }

        mouseWheel = Input.GetAxis("Mouse ScrollWheel");
        cameraChange = Input.GetKeyDown(MovementControlMap.CameraChange);
        increase_far_plane = Input.GetKeyDown(MovementControlMap.IncreaseFarPlane);
        decrease_far_plane = Input.GetKeyDown(MovementControlMap.DecreaseFarPlane);
    }
}
