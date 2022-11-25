using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;
using static SimulationManager;
using static ThirdPersonMovement;

public class Server
{
    public bool server_crash = false;

    public int observations_sent = 0;

    public int actions_received = 0;

    public bool first_observation_sent = false;

    AsyncCallback read_callback = null;
    AsyncCallback write_callback = null;

    byte[] receive_buffer;

    public NetworkStream stream = null;

    public TcpClient client;

    [Serializable]
    public struct ServerConfig
    {
        public LearningConfig learningConfig;
        public RoverConfig roverConfig;
        public EnvironmentConfig envConfig;
    }

    [Serializable]
    struct MessageType
    {
        public string msgType;
    }

    [Serializable]
    public struct JsonMessage<T>
    {
        public string msgType;
        public T payload;
        public bool is_overriden;
    }

    [Serializable]
    public struct ActionSpaceConfig
    {
        public int dimensions;
    }

    [Serializable]
    public struct LearningConfig
    {
        public ActionSpaceConfig actionSpaceConfig;
    }

    [Serializable]
    public struct RoverConfig
    {
        public CameraConfig camConfig;
        public StructureConfig structureConfig;
    }

    [Serializable]
    public struct CameraConfig
    {
        public int[] resolution;
        public int fov;
    }

    [Serializable]
    public struct StructureConfig
    {
        [Range(0f, 1.2f)]
        public float ballastMass;   // rover spec, 6x200g masses
        public float totalBuoyantForce;
    }

    [Serializable]
    public struct EnvironmentConfig
    {
        public FaunaConfig faunaConfig;
    }

    [Serializable]
    public struct FaunaConfig
    {
        public float spawnTimer;
        public float spawnContainerRatio;
        public BoidGroup[] boidGroups;
        public AIGroup[] aiGroups;
        public float spawnRadius;
    }

    [Serializable]
    public struct Buffers
    {
        public int server_send_buffer_size_kb;
        public int client_receive_buffer_size_kb;
        public int server_receive_buffer_size_kb;
        public int client_send_buffer_size_kb;
    }

    /* Configs/Messages from server */
    public JsonMessage<ServerConfig> json_server_config;

    public JsonMessage<JsonControls> json_rover_controls;

    public JsonMessage<ResetEpisode> json_reset_episode;

    public JsonMessage<EndSimulation> json_end_simulation;

    public JsonMessage<EndSimulation> json_server_model_ready;

    public Server()
    {
        read_callback = ProcessData;
        write_callback = OnImageWrite;
    }

    private void OnImageWrite(IAsyncResult result)
    {
    }

    public bool IsTcpGood()
    {
        return (client != null) && (stream != null);
    }

    public void Disconnect()
    {
        client.Close();
        stream = null;
        server_crash = true;
        return;
    }

    public async Task<Exception> Connect(NetworkConfig network_config, string ip, int port)
    {
        client = new TcpClient
        {
            ReceiveBufferSize = network_config.buffers.client_receive_buffer_size_kb * 1024,
            SendBufferSize = network_config.buffers.client_send_buffer_size_kb * 1024
        };

        receive_buffer = new byte[client.ReceiveBufferSize];

        try
        {
            await client.ConnectAsync(ip, port);
            stream = client.GetStream();
            return null;
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            return ex;
        }
    }

    public async Task<GameEvent> ContinueRead()
    {
        try
        {
            while (IsTcpGood())
            {
                int bytes_read = await stream.ReadAsync(receive_buffer, 0, receive_buffer.Length);

                if (bytes_read == 0)
                {
                    await Task.Yield();
                    break;
                }

                string jsonStr = Encoding.Default.GetString(receive_buffer);
                Array.Clear(receive_buffer, 0, receive_buffer.Length);

                if (jsonStr != null)
                {
                    Debug.Log("Received: " + jsonStr);
                    MessageType message = JsonUtility.FromJson<MessageType>(jsonStr);

                    try
                    {
                        switch (message.msgType)
                        {
                            case "process_server_config":
                                json_server_config = JsonUtility.FromJson<JsonMessage<ServerConfig>>(jsonStr);
                                json_server_config.is_overriden = true;
                                break;
                            case "reset_episode":
                                json_reset_episode = JsonUtility.FromJson<JsonMessage<ResetEpisode>>(jsonStr);
                                json_reset_episode.is_overriden = true;
                                break;
                            case "end_simulation":
                                json_end_simulation = JsonUtility.FromJson<JsonMessage<EndSimulation>>(jsonStr);
                                json_end_simulation.is_overriden = true;
                                break;
                            case "receive_json_controls":
                                json_rover_controls = JsonUtility.FromJson<JsonMessage<JsonControls>>(jsonStr);
                                json_rover_controls.is_overriden = true;
                                break;
                            default:
                                break;
                        }
                    }

                    catch (Exception e)
                    {
                        Debug.LogException(e);
                    }
                }
            }
        }
        
        catch(Exception e)
        {
            Debug.LogException(e);
            Disconnect();
            return null;
        }

        return null;
    }

    public async Task SendDataAsync<T>(T data)
    {
        try
        {

            string json_str = JsonUtility.ToJson(data);
            Debug.Log("Sending: " + json_str);

            if (SimulationManager._instance.debug_config.msgType.Length > 0)
            {
                if (SimulationManager._instance.debug_config.payload.save_sent_packets)
                {
                    await File.WriteAllTextAsync(SimulationManager._instance.debug_config.payload.packet_sent_dir + "sent_data_" + observations_sent.ToString() + ".json", json_str);
                }
            }

            byte[] _packet = Encoding.UTF8.GetBytes(json_str);
            await stream.WriteAsync(_packet, 0, _packet.Length);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            Disconnect();
        }
    }

    [Serializable]
    public struct JsonControls
    {
        public float lateralThrust;
        public float forwardThrust;
        public float verticalThrust;
        public float pitchThrust;
        public float yawThrust;
        public float rollThrust;
        public float depthHoldMode;
    }

    [Serializable]
    public struct ResetEpisode
    {
        public bool reset_episode;
    }

    [Serializable]
    public struct EndSimulation
    {
        public bool end_simulation;
    }

    [Serializable]
    public struct ServerModelReady
    {
        public bool model_ready;
    }

    private void ProcessData(IAsyncResult result)
    {
        try
        {
            int _byteLength = stream.EndRead(result);

            if (_byteLength <= 0)
            {
                if (client.Connected)
                {
                    client.Close();
                    return;
                }
            }

            byte[] _data = new byte[_byteLength];
            Array.Copy(receive_buffer, _data, _byteLength);          
        }
        catch (Exception e)
        {
            Debug.LogError(e);
            Disconnect();
            return;
        }
    }
}
