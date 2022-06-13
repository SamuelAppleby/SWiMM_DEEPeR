using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class Server 
{
    private bool save_packet_data;

    private string network_config_dir;

    private string file_write_dir = "C:/Users/sambu/Documents/CodeBases/RLSimulation/network/logs/sent_packets/";

    public int sequence_num = 1;

    public bool first_observation_sent = false;
    public bool ready_to_send = false;

    AsyncCallback read_callback = null;
    AsyncCallback write_callback = null;

    Byte[] receive_buffer;
    public string ip = "127.0.0.1";
    public int port = 60260;

    public NetworkStream stream = null;

    public TcpClient client;

    private Network_Config network_config;

    [Serializable]
    public struct Buffers
    {
        public int server_send_buffer_size_kb;
        public int client_receive_buffer_size_kb;
        public int server_receive_buffer_size_kb;
        public int client_send_buffer_size_kb;
    }

    [Serializable]
    public struct Network_Config
    {
        public string host;
        public int port;
        public Buffers buffers;
    }

    [Serializable]
    public struct JsonMessage<T>
    {
        public T payload;
        public bool is_overridden;

        public void Reset()
        {
            is_overridden = false;
        }
    }

    [Serializable]
    public struct ServerConfig
    {
        public RoverConfig roverConfig;
        public EnvironmentConfig envConfig;
    }

    [Serializable]
    public struct RoverConfig
    {
        public CameraConfig camConfig;
    }

    [Serializable]
    public struct CameraConfig
    {
        public int fov;
    }

    [Serializable]
    public struct EnvironmentConfig
    {
        public FogConfig fogConfig;
    }

    [Serializable]
    public struct FogConfig
    {
        public float fogStart;
        public float fogEnd;
        public bool fogOn;
    }

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

    public void ProcessNetworkConfig()
    {
        using (StreamReader r = new StreamReader(network_config_dir))
        {
            string json = r.ReadToEnd();
            network_config = JsonUtility.FromJson<Network_Config>(json);
        }

        ip = network_config.host;
        port = network_config.port;
    }

    public async Task Connect()
    {
#if UNITY_EDITOR
        network_config_dir = "../network/data/network_config.json";
#else
        network_config_dir = "../../../network/data/network_config.json";
#endif

        ProcessNetworkConfig();

        if (/*save_packet_data*/true)
        {
            DirectoryInfo di = new DirectoryInfo(file_write_dir);

            FileInfo[] files = di.GetFiles();
            foreach (FileInfo file in files)
            {
                file.Delete();
            }
        }

        client = new TcpClient
        {
            ReceiveBufferSize = network_config.buffers.client_receive_buffer_size_kb * 1024,
            SendBufferSize = network_config.buffers.client_send_buffer_size_kb * 1024
        };

        receive_buffer = new Byte[client.ReceiveBufferSize];

        try
        {
            await client.ConnectAsync(ip, port);
            stream = client.GetStream();
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
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
        }

    }

    public async void SendDataAsync<T>(T data)
    {
        try
        {
            string json_str = JsonUtility.ToJson(data);
            Debug.Log("Sending: " + json_str);

            if (/*save_packet_data*/true)
            {
                await File.WriteAllTextAsync(file_write_dir + "sent_data_" + sequence_num.ToString() + ".json", json_str);
            }

            byte[] _packet = Encoding.UTF8.GetBytes(json_str);
            Debug.Log("Packet length: " + _packet.Length);
            stream.BeginWrite(_packet, 0, _packet.Length, write_callback, null);
        }
        catch(Exception e)
        {
            Debug.Log($"Error sending data to server via TCP: {e}");
        }
    }

    [Serializable]
    struct MessageType
    {
        public string msgType;
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
        int _byteLength = stream.EndRead(result);

        try
        {
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
                    client.Close();
                    return;
                }
                else
                {
                    ready_to_send = true;
                }

                //byte[] msg = System.Text.Encoding.ASCII.GetBytes(data);

                //// Send back a response.
                //stream.Write(msg, 0, msg.Length);
                //Console.WriteLine("Sent: {0}", data);
            }   
        }
        catch (Exception e)
        {
            if(e != null)
            {
                Debug.LogError(e);
            }

            if (client.Connected)
            {
                client.Close();
                return;
            }

            return;
        }

        stream.BeginRead(receive_buffer, 0, receive_buffer.Length, read_callback, null);
    }
}
