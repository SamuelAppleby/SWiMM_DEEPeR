using System;
using UnityEngine;

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
public class ROVControls
{
    public Vector3 LinearThrustStrength = new Vector3(10, 10, 10);
    public Vector3 AngularThrustStrength = new Vector3(10, 10, 10);

    public AnimationCurve SlopeCurveModifier = new AnimationCurve(new Keyframe(-90.0f, 1.0f), new Keyframe(0.0f, 1.0f), new Keyframe(90.0f, 0.0f));

    public float minFov = 15f;
    public float maxFov = 120f;
    public float sensitivity = 20f;

    [HideInInspector] public bool decrease_far_plane = false;

    public void Update(bool in_manual_mode, ROVController mov)
    {
        /* Get manual input when not using tcp */
        if (in_manual_mode)
        {
            mov.linear_force_to_be_applied += new Vector3(Input.GetAxisRaw(MovementControlMap.StrafeKey),
                Input.GetKey(MovementControlMap.RiseKey) ? 1 : Input.GetKey(MovementControlMap.FallKey) ? -1 : 0, Input.GetAxisRaw(MovementControlMap.ForwardKey));

            mov.angular_force_to_be_applied += new Vector3(Input.GetAxisRaw(MovementControlMap.PitchKey),
                Input.GetAxisRaw(MovementControlMap.YawKey), Input.GetKey(MovementControlMap.RollPositive) ? 1 : Input.GetKey(MovementControlMap.RollNegative) ? -1 : 0);

            if (Input.GetKeyDown(MovementControlMap.HoverKey))
            {
                mov.ToggleDepthHoldMode();
            }
        }

        if (Input.GetAxis("Mouse ScrollWheel") != 0)
        {
            mov.OnMouseWheelScroll(Input.GetAxis("Mouse ScrollWheel"));
        }

        if(Input.GetKeyDown(MovementControlMap.CameraChange))
        {
            mov.OnChangeCamera();
        }

        if (Input.GetKeyDown(MovementControlMap.IncreaseFarPlane))
        {
            mov.ChangeFarPlane(100);
        }

        if (Input.GetKeyDown(MovementControlMap.DecreaseFarPlane))
        {
            mov.ChangeFarPlane(-100);
        }
    }
}
