import threading
import socket
import socketserver
import sys

REMOTE_HOST = "seppe.net"
REMOTE_PORT = 7778

LOCAL_HOST = "0.0.0.0"
LOCAL_PORT = 8080

class MyTCPHandler(socketserver.StreamRequestHandler):

    def handle(self):
        channels = ",".join(sys.argv[1:])

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((REMOTE_HOST, REMOTE_PORT))
        client.send((channels + "\n").encode('utf-8'))
        client_mf = client.makefile()

        while True:
            try:
                line = client_mf.readline()
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            except (Exception, IOError) as e:
                break

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Please provide a list of Twitch channels")
        print("E.g. python local_proxy.py ChannelOne ChannelTwo")

    else:
        with ThreadedTCPServer((LOCAL_HOST, LOCAL_PORT), MyTCPHandler) as server:
            print(f"Remote {REMOTE_HOST}:{REMOTE_PORT} forwarded to {LOCAL_HOST}:{LOCAL_PORT}")
            print("Press CTRL+C to stop")
            server.serve_forever()