using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using static ThirdPersonMovement;

public class Server 
{
    private string network_config_dir;
    private string debug_config_dir;

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

    [Serializable]
    public struct DebugConfig
    {
        public bool save_images;
        public string image_dir;
        public bool save_sent_packets;
        public string packet_sent_dir;
    }

    [Serializable]
    public struct Buffers
    {
        public int server_send_buffer_size_kb;
        public int client_receive_buffer_size_kb;
        public int server_receive_buffer_size_kb;
        public int client_send_buffer_size_kb;
    }

    [Serializable]
    public struct NetworkConfig
    {
        public string host;
        public int port;
        public Buffers buffers;
    }

    /* Local configs for reading */
    private JsonMessage<DebugConfig> debug_config;

    private JsonMessage<NetworkConfig> network_config;

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

    public void ProcessConfig<T>(ref JsonMessage<T> config, string dir)
    {
        using (StreamReader r = new StreamReader(dir))
        {
            string json = r.ReadToEnd();
            config.payload = JsonUtility.FromJson<T>(json);
            config.is_overridden = true;
        }
    }

    public void Clear_Cache()
    {
        if (debug_config.is_overridden)
        {
            DirectoryInfo di = new DirectoryInfo(debug_config.payload.packet_sent_dir);

            if (di.Exists)
            {
                FileInfo[] files = di.GetFiles();
                foreach (FileInfo file in files)
                {
                    file.Delete();
                }
            }
            else
            {
                System.IO.Directory.CreateDirectory(di.FullName);
            }

            di = new DirectoryInfo(debug_config.payload.image_dir);

            if (di.Exists)
            {
                FileInfo[] files = di.GetFiles();
                foreach (FileInfo file in files)
                {
                    file.Delete();
                }
            }
            else
            {
                System.IO.Directory.CreateDirectory(di.FullName);
            }
        }
    }

    public async Task Connect()
    {
#if UNITY_EDITOR
        debug_config_dir = "../Configs/data/client_debug_config.json";
        network_config_dir = "../Configs/data/network_config.json";
#else
        debug_config_dir = "../../../Configs/data/client_debug_config.json";
        network_config_dir = "../../../Configs/data/network_config.json";
#endif

        ProcessConfig<DebugConfig>(ref debug_config, debug_config_dir);
        ProcessConfig<NetworkConfig>(ref network_config, network_config_dir);

        Clear_Cache();

        if (network_config.is_overridden)
        {
            client = new TcpClient
            {
                ReceiveBufferSize = network_config.payload.buffers.client_receive_buffer_size_kb * 1024,
                SendBufferSize = network_config.payload.buffers.client_send_buffer_size_kb * 1024
            };

            receive_buffer = new Byte[client.ReceiveBufferSize];

            try
            {
                await client.ConnectAsync(network_config.payload.host, network_config.payload.port);
                stream = client.GetStream();
                ContinueRead();
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
            }
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

            if (debug_config.is_overridden && debug_config.payload.save_sent_packets)
            {
                await File.WriteAllTextAsync(debug_config.payload.packet_sent_dir + "sent_data_" + sequence_num.ToString() + ".json", json_str);
                DataToSend<Telemetary_Data>? obj = data as DataToSend<Telemetary_Data>?;

                if (obj != null && debug_config.is_overridden & debug_config.payload.save_images)
                {
                    File.WriteAllBytes(debug_config.payload.image_dir + "sent_image" + sequence_num.ToString() + ".jpg", obj.Value.payload.jpg_image);
                }
            }

            byte[] _packet = Encoding.UTF8.GetBytes(json_str);
            stream.BeginWrite(_packet, 0, _packet.Length, write_callback, null);
        }
        catch(Exception e)
        {
            Debug.Log($"Error sending data to server via TCP: {e}");
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
