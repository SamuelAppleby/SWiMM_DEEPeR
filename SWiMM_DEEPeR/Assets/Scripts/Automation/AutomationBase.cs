using System;
using System.Collections.Generic;
using UnityEngine;

public class AutomationBase : MonoBehaviour
{
    protected float action_timer;
    protected float TIME_TO_ACT;

    [HideInInspector]
    public List<Enums.E_Automation_Actions> current_actions;

    protected MainMenu menu_ref;

    protected virtual void Awake()
    {
        TIME_TO_ACT = 5.0f;
        action_timer = 0.0f;
        current_actions = new List<Enums.E_Automation_Actions>();
    }

    protected virtual void Update()
    {
        if (current_actions.Count > 0)
        {
            action_timer -= Time.deltaTime;

            if (action_timer < 0.0f)
            {
                switch (current_actions[0])
                {
                    case Enums.E_Automation_Actions.TO_OPTIONS:
                        menu_ref.options_button.onClick.Invoke();
                        break;
                    case Enums.E_Automation_Actions.SERVER_CONNECT:
                        menu_ref.Connect(SimulationManager._instance.network_config.host, SimulationManager._instance.network_config.port);
                        break;
                    case Enums.E_Automation_Actions.TO_MENU:
                        menu_ref.back_button.onClick.Invoke();
                        break;
                    case Enums.E_Automation_Actions.START_SERVER_CONTROL:
                        menu_ref.PlayGame(1);
                        break;
                    default:
                        break;
                }

                current_actions.RemoveAt(0);
                action_timer = TIME_TO_ACT;
            }
        }
    }

    public virtual void OnSceneChanged(Enums.E_SceneIndices to)
    {
        switch (to)
        {
            case Enums.E_SceneIndices.MAIN_MENU:
                menu_ref = FindObjectOfType<MainMenu>();
                break;
            default:
                break;
        }

        action_timer = TIME_TO_ACT;
    }

    public virtual void OnServerConnectionResponse(Exception e)
    {
        if (e == null)
        {
            current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                Enums.E_Automation_Actions.TO_MENU
            });

        }
        /* Connection failed the first time, might be pip installing, try again (recursive) */
        else
        {
            current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                    Enums.E_Automation_Actions.SERVER_CONNECT
            });
        }

        action_timer = TIME_TO_ACT;
    }
}
