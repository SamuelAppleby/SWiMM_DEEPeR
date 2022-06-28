using System;
using UnityEngine;

static class GlobalControlMap
{
    public static KeyCode QuitKey = KeyCode.Escape;
    public static KeyCode ChangeWindowKey = KeyCode.P;
    public static KeyCode ReloadKey = KeyCode.RightBracket;
    public static KeyCode CursorLock = KeyCode.L;
    public static KeyCode WaterToggle = KeyCode.F10;
    public static KeyCode LightingToggle = KeyCode.F11;
    public static KeyCode ResetNPCs = KeyCode.F12;
}

[Serializable]
public class GlobalControlSettings
{
    [HideInInspector] public bool quitting = false;
    [HideInInspector] public bool changeWindow = false;
    [HideInInspector] public bool reload_scene = false;
    [HideInInspector] public bool cursor_change = false;
    [HideInInspector] public bool water_toggle = false;
    [HideInInspector] public bool volumetric_lighting_toggle = false;
    [HideInInspector] public bool reset_ncps = false;

    public void Update(bool manual_controls)
    {
        quitting = Input.GetKeyUp(GlobalControlMap.QuitKey);
        changeWindow = Input.GetKeyDown(GlobalControlMap.ChangeWindowKey);

        if (manual_controls)
        {
            reload_scene = Input.GetKeyDown(GlobalControlMap.ReloadKey);
        }
        else if (SimulationManager._instance.server.global_command.is_overridden)
        {
            reload_scene = SimulationManager._instance.server.global_command.payload.reset_episode;
            quitting = SimulationManager._instance.server.global_command.payload.end_simulation;
            SimulationManager._instance.server.global_command.Reset();
        }

        cursor_change = Input.GetKeyDown(GlobalControlMap.CursorLock);
        water_toggle = Input.GetKeyDown(GlobalControlMap.WaterToggle);
        volumetric_lighting_toggle = Input.GetKeyDown(GlobalControlMap.LightingToggle);
        reset_ncps = Input.GetKeyDown(GlobalControlMap.ResetNPCs);
    }
}
