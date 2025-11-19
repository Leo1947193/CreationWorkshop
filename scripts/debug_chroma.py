import chromadb
from chromadb.telemetry import Telemetry

print('client version:', chromadb.__version__)
print('telemetry class:', Telemetry)
