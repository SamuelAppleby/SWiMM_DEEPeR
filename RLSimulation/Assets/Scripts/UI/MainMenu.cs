using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Audio;
using static Server;
using UnityEngine.EventSystems;
using Newtonsoft.Json;

public class MainMenu : MonoBehaviour
{
    public AudioMixer audio_mixer;

    [SerializeField]
    private string simulation_name;

    [SerializeField]
    private GameObject options_menu;

    [SerializeField]
    private GameObject main_menu;

    [SerializeField]
    private TMP_InputField ip_addr;

    [SerializeField]
    private TMP_InputField port;

    [SerializeField]
    private Button connect_button;
    private TextMeshProUGUI connect_text;

    [SerializeField]
    private Button training_button;
    private TextMeshProUGUI training_button_text;

    [SerializeField]
    private Button nn_button;
    private TextMeshProUGUI nn_button_text;

    [SerializeField]
    private TextMeshProUGUI network_message;

    [SerializeField]
    private Image network_image;

    [SerializeField]
    private Sprite healthy_network;

    [SerializeField]
    private Sprite unhealthy_network;

    [SerializeField]
    private TextMeshProUGUI working_directory_text;

    public Slider volume_slider;

    public GraphicRaycaster ui_ray;

    public Button options_button;

    public Button back_button;

    public void OnAwaitingTraining()
    {
        training_button_text.color = Color.white;
        training_button.interactable = true;
        training_button.image.enabled = true;
        nn_button_text.color = Color.white;
        nn_button.interactable = true;
        nn_button.image.enabled = true;
    }

    void OnEnable()
    {
        connect_button.onClick.AddListener(() => Connect(ip_addr.text, int.Parse(port.text)));
    }

    void OnDisable()
    {
        connect_button.onClick.RemoveAllListeners();
    }

    private void Start()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        main_menu.SetActive(true);
        options_menu.SetActive(false);
        ip_addr.text = SimulationManager._instance.network_config.host;
        port.text = SimulationManager._instance.network_config.port.ToString();
        connect_text = connect_button.GetComponentInChildren<TextMeshProUGUI>();
        training_button_text = training_button.GetComponentInChildren<TextMeshProUGUI>();
        nn_button_text = nn_button.GetComponentInChildren<TextMeshProUGUI>();
        working_directory_text.text = System.IO.Directory.GetCurrentDirectory();
        ui_ray.enabled = true;
    }

    private void Awake()
    {
        float volume = 0F;
        audio_mixer.GetFloat("Volume", out volume);
        volume_slider.value = volume;
    }

    public void PlayGame(int level)
    {
        if ((Enums.E_LevelType)level != Enums.E_LevelType.MANUAL)
        {
            SimulationManager._instance.server.json_str_obsv = JsonConvert.SerializeObject(new DataToSend
            {
                msg_type = "training_ready"
            });

            SimulationManager._instance.processing_obj.SetActive(true);
            SimulationManager._instance.processing_obj.GetComponentInChildren<TextMeshProUGUI>().text = "Model initialising...";
            ui_ray.enabled = false;
        }

        else
        {
            SimulationManager._instance.MoveToScene(Enums.E_SceneIndices.SIMULATION, true);
        }
    }

    public void TerminateApplication()
    {
        SimulationManager._instance.QuitApplication();
    }

    public void OnServerConnectionResponse(Exception e)
    {
        if(Enums.protocol_mapping[SimulationManager._instance.network_config.protocol] == Enums.E_Protocol.TCP)
        {
            network_message.text = e != null ? "Failed to connect to: " + ip_addr.text + ":" + port.text : "Succcessfully connected to: " + ip_addr.text + ":" + port.text;
            network_image.sprite = e != null ? unhealthy_network : healthy_network;
            ip_addr.interactable = e != null;
            port.interactable = e != null;
            connect_button.image.color = e != null ? Color.white : Color.green;
            connect_text.text = e != null ? "Connect" : "Connected";
            connect_button.interactable = e != null;
        }

        EventSystem.current.SetSelectedGameObject(null);
    }

    public void OnServerConfigReceived(JsonMessage param)
    {
        if (Enums.protocol_mapping[SimulationManager._instance.network_config.protocol] == Enums.E_Protocol.UDP)
        {
            network_message.text = "Succcessfully connected to: " + ip_addr.text + ":" + port.text;
            network_image.sprite = healthy_network;
            ip_addr.interactable = false;
            port.interactable = false;
            connect_button.image.color = Color.green;
            connect_text.text = "Connected";
            connect_button.interactable = false;
            EventSystem.current.SetSelectedGameObject(null);
        }
    }

    public void Connect(string ip, int port)
    {
        EventMaster._instance.server_connecting_event.Raise(ip, port);
    }

    public void ChangeToOptions(bool to_options)
    {
        main_menu.SetActive(!to_options);
        options_menu.SetActive(to_options);

        if (to_options)
        {
            network_message.text = "";
        }
    }

    public void SetVolume(float volume)
    {
        audio_mixer.SetFloat("Volume", volume);
    }
}
