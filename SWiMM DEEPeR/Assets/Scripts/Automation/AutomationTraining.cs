using System;
using System.Collections.Generic;
using UnityEngine;

public class AutomationTraining : AutomationBase
{
    protected override void Awake()
    {
        base.Awake();
    }

    protected override void Update()
    {
        base.Update();
    }

    public override void OnSceneChanged(Enums.E_SceneIndices to)
    {
        base.OnSceneChanged(to);

        switch (to)
        {
            case Enums.E_SceneIndices.MAIN_MENU:
                current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                    Enums.E_Automation_Actions.TO_OPTIONS,
                    Enums.E_Automation_Actions.SERVER_CONNECT
                });
                break;
            default:
                break;
        }
    }

    public override void OnServerConnectionResponse(Exception e)
    {
        base.OnServerConnectionResponse(e);

        if (e == null)
        {
            current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                Enums.E_Automation_Actions.START_TRAINING
            });
        }
    }
}
