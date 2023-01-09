using System;
using Unity.VisualScripting;
using UnityEngine;

static class GlobalControlMap
{
    public static KeyCode QuitKey = KeyCode.Escape;
    public static KeyCode ChangeWindowKey = KeyCode.P;
    public static KeyCode ReloadKey = KeyCode.R;
    public static KeyCode CursorLock = KeyCode.L;
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
                SimulationManager._instance.EpisodeReset(true);
            }
        }

        if (Input.GetKeyUp(GlobalControlMap.QuitKey))
        {
            switch (SimulationManager._instance.current_scene_index)
            {
                case Enums.E_SceneIndices.MAIN_MENU:
                    Utils.QuitApplication();
                    break;
                case Enums.E_SceneIndices.SIMULATION:
                    SimulationManager._instance.MoveToScene(Enums.E_SceneIndices.MAIN_MENU);
                    break;
                default:
                    break;

            }
        }

        if (Input.GetKeyDown(GlobalControlMap.ChangeWindowKey))
        {
            SimulationManager._instance.IndexWindow();
        }

        if (Input.GetKeyDown(GlobalControlMap.CursorLock))
        {
            SimulationManager._instance.IndexCursor();
        }

        if (Input.GetKeyDown(GlobalControlMap.ResetNPCs))
        {
            SimulationManager._instance.ResetNPCs();
        }
    }
}
