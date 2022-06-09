using System;
using UnityEngine;
using static Server;
using static SimulationManager;

static class GlobalControlMap
{
    public static KeyCode QuitKey = KeyCode.Escape;
    public static KeyCode ChangeWindowKey = KeyCode.P;
    public static KeyCode ReloadKey = KeyCode.RightBracket;
}

[Serializable]
public class GlobalControlSettings
{
    [HideInInspector] public bool quitting = false;
    [HideInInspector] public bool changeWindow = false;
    [HideInInspector] public bool reload_scene = false;

    public void Update(bool use_server)
    {
        quitting = Input.GetKeyDown(GlobalControlMap.QuitKey);
        changeWindow = Input.GetKeyDown(GlobalControlMap.ChangeWindowKey);

        if (!use_server)
        {
            reload_scene = Input.GetKey(GlobalControlMap.ReloadKey);
        }
        else if (SimulationManager._instance.server.global_command.is_overridden)
        {
            reload_scene = SimulationManager._instance.server.global_command.payload.reset_episode;
            quitting = SimulationManager._instance.server.global_command.payload.end_simulation;
            SimulationManager._instance.server.global_command.Reset();
        }
    }
}
