import subprocess
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
import datetime

# Generate our key
key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Write our key to disk for safe keeping
with open("key.pem", "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

# Various details about who we are.
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"California"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Stress Sensor App"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"192.168.19.115"),
])
import ipaddress

ip_str = subprocess.check_output(['python', '-c', 'import socket; print(socket.gethostbyname(socket.gethostname()))']).decode().strip()

cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    # Our certificate will be valid for 10 days
    datetime.datetime.utcnow() + datetime.timedelta(days=10)
).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName(u"localhost"),
        x509.IPAddress(ipaddress.ip_address(ip_str))
    ]),
    critical=False,
# Sign our certificate with our private key
).sign(key, hashes.SHA256())

# Write our certificate out to disk.
with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("Successfully generated cert.pem and key.pem")
