using System;
using UnityEngine;
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

    public void OverrideGlobalControls(GlobalCommand command)
    {
        reload_scene = command.reset_episode;
    }

    public void Update(bool useServer)
    {
        if (!useServer)
        {
            reload_scene = Input.GetKey(GlobalControlMap.ReloadKey);
        }

        quitting = Input.GetKeyDown(GlobalControlMap.QuitKey);
        changeWindow = Input.GetKeyDown(GlobalControlMap.ChangeWindowKey);
    }
}
