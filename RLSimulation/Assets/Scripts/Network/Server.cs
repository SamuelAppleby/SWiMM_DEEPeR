using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using static SimulationManager;
using static ThirdPersonMovement;

public class Server
{
    public bool server_crash = false;
    public bool connected = false;

    public int sequence_num = 1;

    public bool first_observation_sent = false;
    public bool ready_to_send = false;

    AsyncCallback read_callback = null;
    AsyncCallback write_callback = null;

    Byte[] receive_buffer;

    public NetworkStream stream = null;

    public TcpClient client;

    [Serializable]
    public struct ServerConfig
    {
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
        public bool is_overridden;

        public void Reset()
        {
            is_overridden = false;
        }
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
        public FogConfig fogConfig;
        public FaunaConfig faunaConfig;
    }

    [Serializable]
    public struct FogConfig
    {
        public float fogDensity;
        public int[] fogColour;
        public bool fogOn;
    }

    [Serializable]
    public struct FaunaConfig
    {
        public float spawnTimer;
        public float spawnContainerRatio;
        public AIGroup[] aiGroups;
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
    public JsonMessage<ServerConfig> server_config;

    public JsonMessage<JsonControls> rover_controls;

    public JsonMessage<GlobalCommand> global_command;

    public Server()
    {
        read_callback = ProcessData;
        write_callback = OnImageWrite;
    }

    private void OnImageWrite(IAsyncResult result)
    {
        sequence_num++;
    }

    public bool IsTcpGood()
    {
        return (client != null) && (stream != null);
    }

    public void Disconnect()
    {

        client.Close();
        stream = null;
        connected = false;
        server_crash = true;
        return;
    }

    public async Task<Exception> Connect(NetworkConfig network_config)
    {
        client = new TcpClient
        {
            ReceiveBufferSize = network_config.buffers.client_receive_buffer_size_kb * 1024,
            SendBufferSize = network_config.buffers.client_send_buffer_size_kb * 1024
        };

        receive_buffer = new Byte[client.ReceiveBufferSize];

        try
        {
            await client.ConnectAsync(network_config.host, network_config.port);
            stream = client.GetStream();
            connected = true;
            ContinueRead();
            return null;
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            return ex;
        }
    }

    public void ContinueRead()
    {
        try
        {
            if (IsTcpGood())
            {
                stream.BeginRead(receive_buffer, 0, receive_buffer.Length, read_callback, null);
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            Disconnect();
        }
    }

    public async void SendDataAsync<T>(T data)
    {
        try
        {

            string json_str = JsonUtility.ToJson(data);
            Debug.Log("Sending: " + json_str);

            if (SimulationManager._instance.debug_config.is_overridden)
            {
                if (SimulationManager._instance.debug_config.payload.save_sent_packets)
                {
                    await File.WriteAllTextAsync(SimulationManager._instance.debug_config.payload.packet_sent_dir + "sent_data_" + sequence_num.ToString() + ".json", json_str);
                }

                if (SimulationManager._instance.debug_config.payload.save_images)
                {
                    DataToSend<Telemetary_Data>? obj = data as DataToSend<Telemetary_Data>?;

                    if(obj != null)
                    {
                        File.WriteAllBytes(SimulationManager._instance.debug_config.payload.image_dir + "sent_image" + sequence_num.ToString() + ".jpg", obj.Value.payload.jpg_image);
                    }
                }
            }

            byte[] _packet = Encoding.UTF8.GetBytes(json_str);
            stream.BeginWrite(_packet, 0, _packet.Length, write_callback, null);
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
        public float forwardThrust;
        public float verticalThrust;
        public float yRotation;
    }

    [Serializable]
    public struct GlobalCommand
    {
        public bool reset_episode;
        public bool end_simulation;
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

            string jsonStr = System.Text.Encoding.Default.GetString(_data);

            if (jsonStr != null)
            {
                Debug.Log("Received: " + jsonStr);

                MessageType message = JsonUtility.FromJson<MessageType>(jsonStr);

                try
                {
                    switch (message.msgType)
                    {
                        case "process_server_config":
                            server_config = JsonUtility.FromJson<JsonMessage<ServerConfig>>(jsonStr);
                            server_config.is_overridden = true;
                            break;
                        case "receive_json_controls":
                            rover_controls = JsonUtility.FromJson<JsonMessage<JsonControls>>(jsonStr);
                            rover_controls.is_overridden = true;
                            break;
                        case "global_message":
                            global_command = JsonUtility.FromJson<JsonMessage<GlobalCommand>>(jsonStr);
                            global_command.is_overridden = true;
                            break;
                        default:
                            break;
                    }
                }

                catch (Exception e)
                {
                    Debug.LogException(e);
                }


                if (global_command.is_overridden && global_command.payload.end_simulation)
                {
                    Disconnect();
                    return;
                }
                else
                {
                    ready_to_send = true;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError(e);
            Disconnect();
            return;
        }

        stream.BeginRead(receive_buffer, 0, receive_buffer.Length, read_callback, null);
    }
}
