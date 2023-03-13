using System;
using System.Collections.Generic;
using UnityEngine;
using static Enums;

public class AutomationImageSampling : AutomationBase
{
    protected override void Awake()
    {
        base.Awake();
    }

    protected override void Update()
    {
        base.Update();
    }

    public override void OnSceneChanged(E_SceneIndices to)
    {
        base.OnSceneChanged(to);

        switch (to)
        {
            case E_SceneIndices.MAIN_MENU:
                current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                    E_Automation_Actions.START_SERVER_CONTROL
                });
                break;
            case E_SceneIndices.SIMULATION:
                switch (SimulationManager._instance.game_state)
                {
                    case E_Game_State.IMAGE_SAMPLING:
                        FindObjectOfType<ImageSampling>().enabled = true;
                        break;
                    case E_Game_State.VAE_GEN:
                        FindObjectOfType<VAEImageGeneration>().enabled = true;
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
    }

    public override void OnServerConnectionResponse(Exception e)
    {
        base.OnServerConnectionResponse(e);
    }
}
