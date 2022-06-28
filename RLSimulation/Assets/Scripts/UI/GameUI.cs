using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class GameUI : MonoBehaviour
{
    [SerializeField]
    private GameObject fps_obj;
    private TextMeshProUGUI fps_value_text;

    [SerializeField]
    private GameObject packets_obj;
    private TextMeshProUGUI packes_value_text;

    [SerializeField]
    private GameObject hovering_obj;
    private TextMeshProUGUI hovering_value_text;

    void Start()
    {
        fps_value_text = fps_obj.GetComponent<TextMeshProUGUI>();
        packes_value_text = packets_obj.GetComponent<TextMeshProUGUI>();
        hovering_value_text = hovering_obj.GetComponent<TextMeshProUGUI>();
    }

    void Update()
    {
        fps_value_text.text = SimulationManager._instance.avgFrameRate.ToString();
        packes_value_text.text = (SimulationManager._instance.server.sequence_num - 1).ToString();
        hovering_value_text.text = SimulationManager._instance.rover.GetComponent<ThirdPersonMovement>().m_Hovering.ToString();
    }
}
