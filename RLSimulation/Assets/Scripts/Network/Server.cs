using System;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using static SimulationManager;

public class Server 
{
    public bool first_observation_sent = false;
    public bool ready_to_send = false;

    AsyncCallback read_callback = null;
    AsyncCallback write_callback = null;

    Byte[] bytes = new Byte[256];
    public string ip = "127.0.0.1";
    public int port = 60260;

    public NetworkStream stream = null;

    public TcpClient client;

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
    public struct ConfigOptions
    {
        public CameraConfig camConfig;
        public EnvironmentConfig envConfig;
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

    public JsonMessage<ConfigOptions> server_config;

    public JsonMessage<JsonControls> rover_controls;

    public JsonMessage<GlobalCommand> global_command;

    public Server(ServerInfo info)
    {
        ip = info.ip;
        port = info.port;
    }

    private void OnImageWrite(IAsyncResult result)
    {
    }

    public bool IsTcpGood()
    {
        return (client != null) && (stream != null);
    }

    public async Task Connect()
    {
        read_callback = ProcessData;
        write_callback = OnImageWrite;

        client = new TcpClient
        {
            ReceiveBufferSize = bytes.Length,
            SendBufferSize = bytes.Length
        };

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
                stream.BeginRead(bytes, 0, bytes.Length, read_callback, null);
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }

    }

    public void SendDataAsync(byte[] _packet, AsyncCallback callback)
    {
        try
        {
            if ((client != null) && (stream != null))
            {
                stream.BeginWrite(_packet, 0, _packet.Length, callback, null);
            }
        }
        catch (Exception _ex)
        {
            Debug.Log($"Error sending data to server via TCP: {_ex}");
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
            Array.Copy(bytes, _data, _byteLength);

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
                            server_config = JsonUtility.FromJson<JsonMessage<ConfigOptions>>(jsonStr);
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

        stream.BeginRead(bytes, 0, bytes.Length, read_callback, null);
    }

    public void SendImageData(string data)
    {
        Debug.Log("Sending: " + data);
        SendDataAsync(Encoding.UTF8.GetBytes(data), write_callback);
    }
}
