using System;
using UnityEngine;

static class GlobalControlMap
{
    public static KeyCode QuitKey = KeyCode.Escape;
    public static KeyCode ChangeWindowKey = KeyCode.P;
    public static KeyCode ReloadKey = KeyCode.R;
    public static KeyCode CursorLock = KeyCode.L;
    public static KeyCode WaterToggle = KeyCode.F10;
    public static KeyCode LightingToggle = KeyCode.F11;
    public static KeyCode ResetNPCs = KeyCode.F12;
}

[Serializable]
public class GlobalControlSettings
{
    public void Update(bool manual_controls)
    {
        if (manual_controls)
        {
            if (Input.GetKeyUp(GlobalControlMap.ReloadKey))
            {
                SimulationManager._instance.ResetEpiosde(true);
            }
        }

        if (Input.GetKeyUp(GlobalControlMap.QuitKey))
        {
            SimulationManager._instance.MoveToScene(SimulationManager._instance.current_scene_index == SceneIndices.MAIN_MENU ? SceneIndices.EXIT : SceneIndices.MAIN_MENU);
        }

        if (Input.GetKeyDown(GlobalControlMap.ChangeWindowKey))
        {
            SimulationManager._instance.IndexWindow();
        }

        if (Input.GetKeyDown(GlobalControlMap.CursorLock))
        {
            SimulationManager._instance.IndexWindow();
        }

        if (Input.GetKeyDown(GlobalControlMap.WaterToggle))
        {
            SimulationManager._instance.ToggleWater();
        }

        if (Input.GetKeyDown(GlobalControlMap.LightingToggle))
        {
            SimulationManager._instance.ToggleVolumetricLighting();
        }

        if (Input.GetKeyDown(GlobalControlMap.ResetNPCs))
        {
            SimulationManager._instance.ResetNPCs();
        }
    }
}
