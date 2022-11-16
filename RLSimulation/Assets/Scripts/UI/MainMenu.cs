using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.Audio;

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

    private IEnumerator Start()
    {
        yield return new WaitUntil(() => SimulationManager._instance.IsInitialized);
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        main_menu.SetActive(true);
        options_menu.SetActive(false);
        ip_addr.text = SimulationManager._instance.network_config.payload.host;
        port.text = SimulationManager._instance.network_config.payload.port.ToString();
        connect_text = connect_button.GetComponentInChildren<TextMeshProUGUI>();
        training_button_text = training_button.GetComponentInChildren<TextMeshProUGUI>();
        nn_button_text = nn_button.GetComponentInChildren<TextMeshProUGUI>();
        working_directory_text.text = System.IO.Directory.GetCurrentDirectory();
        ChangeUIServerActive(SimulationManager._instance.server != null && SimulationManager._instance.server.IsTcpGood());
    }

    private void Awake()
    {
        float volume = 0F;
        audio_mixer.GetFloat("Volume", out volume);
        volume_slider.value = volume;
    }

    public void PlayGame(bool manual_controls)
    {
        SimulationManager._instance.MoveToScene(SceneIndices.SIMULATION);
    }

    public void TerminateApplication()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
         Application.Quit();
#endif
    }

    public async void Connect()
    {
        connect_text.text = "Connecting";
        Exception e = await SimulationManager._instance.ConnectToServer(ip_addr.text, int.Parse(port.text));
        if(e != null)
        {
            ResetConnect();
        }
        else
        {
            connect_text.text = "Connected";
            connect_button.image.color = Color.green;
            connect_button.interactable = false;
            ip_addr.interactable = false;
            port.interactable = false;
        }

        ChangeUIServerActive(e == null);
    }

    public void ChangeUIServerActive(bool server_on)
    {
        training_button_text.color = server_on ? Color.white : Color.red;
        training_button.interactable = server_on;
        training_button.image.enabled = server_on;
        nn_button_text.color = server_on ? Color.white : Color.red;
        nn_button.interactable = server_on;
        nn_button.image.enabled = server_on;
        network_message.text = server_on ? "Succcessfully connected to: " + ip_addr.text + ":" + port.text : "Failed to connect to: " + ip_addr.text + ":" + port.text + ":";
        connect_text.text = server_on ? "Connected" : "Connect";
        connect_button.image.color = server_on ? Color.green : Color.white;
        connect_button.interactable = server_on ? false : true;
        ip_addr.interactable = server_on ? false : true;
        port.interactable = server_on ? false : true;
        network_image.sprite = server_on ? healthy_network : unhealthy_network;
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

    public void ResetConnect()
    {
        connect_text.text = "Connect";
        EventSystem.current.SetSelectedGameObject(null);
    }

    public void SetVolume(float volume)
    {
        audio_mixer.SetFloat("Volume", volume);
    }
}
