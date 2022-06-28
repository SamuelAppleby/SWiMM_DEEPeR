using System;
using UnityEngine;
using static Server;

public static class MovementControlMap
{
    public static KeyCode CameraChange = KeyCode.C;
    public static string ForwardKey = "Vertical";
    public static string SidewaysKey = "Horizontal";
    public static KeyCode RiseKey = KeyCode.Space;
    public static KeyCode FallKey = KeyCode.LeftControl;
    public static KeyCode HoverKey = KeyCode.H;
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
    [HideInInspector] public bool hovering = false;
    [HideInInspector] public float mouseWheel = 0.0f;
    [HideInInspector] public bool cameraChange = false;

    public void Update(bool in_manual_mode)
    {
        /* Get manual input when not using tcp */
        if (in_manual_mode)
        {
            movementInputs.y = Input.GetKey(MovementControlMap.RiseKey) ? 1 : Input.GetKey(MovementControlMap.FallKey) ? -1 : 0;
            movementInputs.z = Input.GetAxisRaw(MovementControlMap.ForwardKey);
            rotationInputs.y = Input.GetAxisRaw(MovementControlMap.SidewaysKey);
            hovering = Input.GetKeyDown(MovementControlMap.HoverKey);
        }
        else if (SimulationManager._instance.server.rover_controls.is_overridden)
        {
            movementInputs.y = SimulationManager._instance.server.rover_controls.payload.verticalThrust;
            movementInputs.z = SimulationManager._instance.server.rover_controls.payload.forwardThrust;
            rotationInputs.y = SimulationManager._instance.server.rover_controls.payload.yRotation;

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
        if (hovering)
        {
            audios.hover_noise.Play();
        }

        mouseWheel = Input.GetAxis("Mouse ScrollWheel");
        cameraChange = Input.GetKeyDown(MovementControlMap.CameraChange);
    }
}
