using System;
using UnityEngine;

static class MovementControlMap
{
    public static KeyCode CameraChange = KeyCode.C;
    public static string ForwardKey = "Vertical";
    public static string SidewaysKey = "Horizontal";
    public static KeyCode RiseKey = KeyCode.Space;
    public static KeyCode FallKey = KeyCode.LeftControl;
    public static KeyCode HoverKey = KeyCode.H;
}

[Serializable]
public class ThirdPersonControlSettings
{
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

    public void ReceiveJsonControls(SimulationManager.JsonControls controls)
    {
        movementInputs.y = controls.verticalThrust;
        movementInputs.z = controls.forwardThrust;
        rotationInputs.y = controls.yRotation;
    }

    public void Update(bool useServer)
    {
        /* Get manual input when not using tcp */
        if (!useServer)
        {
            movementInputs.y = Input.GetKey(MovementControlMap.RiseKey) ? 1 : Input.GetKey(MovementControlMap.FallKey) ? -1 : 0;
            movementInputs.z = Input.GetAxisRaw(MovementControlMap.ForwardKey);
            rotationInputs.y = Input.GetAxisRaw(MovementControlMap.SidewaysKey);
            hovering = Input.GetKeyDown(MovementControlMap.HoverKey);
        }

        mouseWheel = Input.GetAxis("Mouse ScrollWheel");
        cameraChange = Input.GetKeyDown(MovementControlMap.CameraChange);
    }
}
