using System;
using UnityEngine;

static class ControlMap
{
    public static KeyCode CameraChange = KeyCode.C;   
    public static KeyCode RiseKey = KeyCode.Space;
    public static KeyCode FallKey = KeyCode.LeftControl;
    public static KeyCode HoverKey = KeyCode.H;
    public static KeyCode QuitKey = KeyCode.Escape;
    public static KeyCode ChangeWindowKey = KeyCode.P;
}

[Serializable]
public class ControlSettings
{
    public float ThrustPower = 8.0f;
    public AnimationCurve SlopeCurveModifier = new AnimationCurve(new Keyframe(-90.0f, 1.0f), new Keyframe(0.0f, 1.0f), new Keyframe(90.0f, 0.0f));

    public float minFov = 15f;
    public float maxFov = 120f;
    public float sensitivity = 20f;

    [HideInInspector] public Vector3 movementInputs;
    [HideInInspector] public Vector3 rotationInputs;
    [HideInInspector] public bool hovering = false;
    [HideInInspector] public bool cameraChange = false;
    [HideInInspector] public float mouseWheel = 0.0f;
    [HideInInspector] public bool quitting = false;
    [HideInInspector] public bool changeWindow = false;

    public void ReceiveJsonControls(ThirdPersonMovement.JsonControls controls)
    {
        movementInputs.y = controls.verticalThrust;
        movementInputs.z = controls.forwardThrust;
        rotationInputs.y = controls.yRotation;
    }

    public void Update(bool useTcp)
    {
        /* Get manual input when not using tcp */
        if (!useTcp)
        {
            hovering = Input.GetKeyDown(ControlMap.HoverKey);
            movementInputs.y = Input.GetKey(ControlMap.RiseKey) ? 1 : Input.GetKey(ControlMap.FallKey) ? -1 : 0;
            movementInputs.z = Input.GetAxisRaw("Vertical");
            rotationInputs.y = Input.GetAxisRaw("Horizontal");
        }

        mouseWheel = Input.GetAxis("Mouse ScrollWheel");
        cameraChange = Input.GetKeyDown(ControlMap.CameraChange);
        quitting = Input.GetKeyDown(ControlMap.QuitKey);
        changeWindow = Input.GetKeyDown(ControlMap.ChangeWindowKey);
    }
}
