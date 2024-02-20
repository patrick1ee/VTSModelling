using Sockets

const HOST = ip"127.0.0.1"
const PORT = 12345
const DEFAULT_RESPONSE = "D"

function handle_recv(sock::TCPSocket)
    while isopen(sock)
        data = read(sock, Char)
        if isempty(data)
            break
        end

        println("Received from client: $data")
    end
end

function handle_send(sock::TCPSocket)
    while isopen(sock)
        write(sock, DEFAULT_RESPONSE)
        println("Sent default to client: $DEFAULT_RESPONSE")
    end
end


function main()
    server = listen(HOST, PORT)
    println("Julia server listening on $HOST:$PORT...")

    while true
        sock = accept(server)
        println("Connection established with client.")

        # Handle client in a separate task
        @async handle_client(sock)
        @async handle_send(sock)
    end
end

main()