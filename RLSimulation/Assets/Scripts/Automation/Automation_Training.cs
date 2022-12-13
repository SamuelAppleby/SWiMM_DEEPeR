using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Automation_Training : Singleton<Automation_Training>
{
    private float action_timer;
    private float TIME_TO_ACT;
    private MainMenu menu_ref;

    private List<Enums.E_Automation_Actions> current_actions;

    protected override void Awake()
    {
        base.Awake();
        _instance.TIME_TO_ACT = 5.0f;
        _instance.action_timer = 0.0f;
        _instance.current_actions = new List<Enums.E_Automation_Actions>();
    }

    void Update()
    {
        if (_instance.current_actions.Count > 0)
        {
            _instance.action_timer -= Time.deltaTime;

            if (_instance.action_timer < 0.0f)
            {
                Debug.Log((_instance.current_actions[0]));
                switch (_instance.current_actions[0])
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
                    case Enums.E_Automation_Actions.START_TRAINING:
                        menu_ref.PlayGame(1);       // 1 assocites to training in Enums
                        break;
                }

                _instance.current_actions.RemoveAt(0);
                _instance.action_timer = _instance.TIME_TO_ACT;
            }
        }
    }

    public void OnSceneChanged(Enums.E_SceneIndices to)
    {
        switch (to)
        {
            case Enums.E_SceneIndices.MAIN_MENU:
                _instance.menu_ref = FindObjectOfType<MainMenu>();
                _instance.current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                    Enums.E_Automation_Actions.TO_OPTIONS,
                    Enums.E_Automation_Actions.SERVER_CONNECT 
                });
                break;
        }

        _instance.action_timer = _instance.TIME_TO_ACT;
    }

    public void OnServerConnectionResponse(Exception e)
    {
        if (e == null)
        {
            _instance.current_actions.AddRange(new List<Enums.E_Automation_Actions> {
                Enums.E_Automation_Actions.TO_MENU,
                Enums.E_Automation_Actions.START_TRAINING
            });

            _instance.action_timer = _instance.TIME_TO_ACT;
        }
        else
        {
            SimulationManager._instance.QuitApplication();
        }
    }
}
