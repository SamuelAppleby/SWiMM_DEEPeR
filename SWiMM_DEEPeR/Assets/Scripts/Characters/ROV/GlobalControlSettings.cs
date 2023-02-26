using System;
using System.IO;
using UnityEngine;

static class GlobalControlMap
{
    public static KeyCode Key_Quit = KeyCode.Escape;
    public static KeyCode Key_Reload = KeyCode.R;
    public static KeyCode Key_Cursor_Lock = KeyCode.L;
    public static KeyCode Key_Screenshot = KeyCode.F12;
    public static KeyCode Key_Reset_NPCs = KeyCode.F12;
}

[Serializable]
public class GlobalControlSettings
{
    public void Update(bool manual_controls)
    {
        if (manual_controls)
        {
            if (Input.GetKeyUp(GlobalControlMap.Key_Reload))
            {
                SimulationManager._instance.EpisodeReset(true);
            }
        }

        if (Input.GetKeyUp(GlobalControlMap.Key_Quit))
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

        if (Input.GetKeyDown(GlobalControlMap.Key_Cursor_Lock))
        {
            SimulationManager._instance.IndexCursor();
        }

        if (Input.GetKeyDown(GlobalControlMap.Key_Reset_NPCs))
        {
            SimulationManager._instance.ResetNPCs();
        }
    }
}
