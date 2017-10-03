'''
import base64
import json
import zlib
from Crypto.Cipher import Blowfish

# API REFERENCE
# https://developer.riotgames.com/api/methods
#
# Lots of information here
# https://developer.riotgames.com/docs/game-constants
#
# Game API
# https://developer.riotgames.com/api/methods#!/933/3233
#
# Match API (this is what we want most)
# https://developer.riotgames.com/api/methods#!/929/3214

API_KEY = '38940d99-0f69-4dfd-a2ea-5e9de11552a3'

encryptionKey = '0mZ6Y7BNxLx0+hnsh8ARaYiVjONapfql'
tokenFile = '../Data/tokens/token'


data = open(tokenFile, 'rb').read()

print ("File size:", len(data))

print (data[:200])


key = base64.b64decode(encryptionKey)

print ()
print (key, "\tfrom\t", encryptionKey)
print (key.decode('base64'))

cipher = Blowfish.new(encryptionKey, Blowfish.MODE_ECB)
msg = cipher.decrypt(data)

print ()
print (msg[:200])


decompressed = zlib.decompress(msg)

print ()
print (decompressed)
'''
'''
import struct
import json
import copy
import gzip
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

ROFL_MAGIC = 'RIOT' + chr(0) * 2

FILENAME = '20130523_0935_HA_PBE1_26967171.rofl'
FILENAME2 = '20130523_0702_HA_PBE1_26961185.rofl'

# With thanks to https://github.com/robertabcd/lol-ob


class Struct(object):
    format = None
    extradata = None

    @classmethod
    def get_extradata(cls, fileobj):
        return [None] * len(cls.get_format(fileobj, None))

    @classmethod
    def get_format(cls, fileobj, extradata):
        return cls.format

    @classmethod
    def read(cls, fh, fileobj, extradata=None):
        format = cls.get_format(fileobj, extradata=extradata)
        f_str = fh.read(struct.calcsize(format))
        res = struct.unpack(format, f_str)
        me = cls()
        me.unpack_tuple(res, fileobj, extradata)
        return me

    def unpack_tuple(self, res, fileobj, extradata):
        for field_name, field_value in zip(self.fields, res):
            custom_func = getattr(self, "unpack_{}".format(field_name), None)
            if custom_func is not None:
                custom_func(field_name, field_value, fileobj, extradata)
            else:
                setattr(self, field_name, field_value)

    def pack_tuple(self, fileobj, extradata):
        out = []
        for field_name in self.fields:
            custom_func = getattr(self, "pack_{}".format(field_name), None)
            if custom_func is not None:
                val = custom_func(field_name, fileobj, extradata)
            else:
                val = getattr(self, field_name)
            out.push(val)
        return out

    def write(self, fh, fileobj, extradata=None):
        out_tuple = self.pack_tuple(fileobj, extradata)
        fh.write(struct.pack(self.get_format(fileobj, extradata), *out_tuple))


class CompositeStruct(Struct):
    @classmethod
    def read(cls, fh, fileobj, extradata=None):
        self = cls()
        for clazz, field in zip(cls.get_format(fileobj), cls.fields):
            setattr(self, field, clazz.read(fh, self, extradata=extradata))
        return self


class CompositeStructList(Struct):
    @classmethod
    def read(cls, fh, fileobj, extradata=None):
        self = cls()
        self.outer = fileobj
        self.data = []
        for clazz, ed in zip(
            cls.get_format(fileobj, extradata=extradata),
            cls.get_extradata(fileobj)
        ):
            self.data.append(clazz.read(fh, self, extradata=ed))
        return self

    def __str__(self):
        return "[{}]".format(', '.join((str(s) for s in self.data)))


class RoflHeader(Struct):
    format = '6s256sHIIIIII'
    fields = [
        'magic', 'signature', 'header_len', 'file_len',
        'metadata_offset', 'metadata_len', 'payload_header_offset',
        'payload_header_len', 'payload_offset'
    ]

    def __str__(self):
        return "<RoflHeader - magic: {}>".format(self.magic)


class RoflMetadata(Struct):
    fields = ['json']

    @classmethod
    def get_format(cls, fileobj, extradata):
        return "{}s".format(fileobj.header.metadata_len)

    def unpack_json(self, field_name, field_value, fileobj, extradata):
        self.json = json.loads(field_value)
        self.json['stats'] = json.loads(self.json['statsJSON'])
        del self.json['statsJSON']

    def pack_json(self, field_name, fileobj, extradata):
        tmpj = copy.deepcopy(self.json)
        tmpj['statsJSON'] = json.dumps(tmpj['stats'])
        del tmpj['stats']
        return json.dumps(tmpj)

    def as_json(self):
        return json.dumps(self.json, indent=4)

    def __str__(self):
        return "<RoflMetadata>"


class RoflPayloadHeader(Struct):
    format = 'QIIIIIIH'
    fields = [
        'game_id', 'game_length', 'keyframe_count', 'chunk_count',
        'end_startup_chunk_id', 'start_game_chunk_id', 'keyframe_interval',
        'encryption_key_length'
    ]

    def __str__(self):
        return "<RoflPayloadHeader - game ID: {} - game length: {} - " + \
            "keyframe count: {} - chunk count: {}>".format(
                self.game_id, self.game_length,
                self.keyframe_count, self.chunk_count
            )


class RoflEncryptionKey(Struct):
    @classmethod
    def get_format(cls, fileobj, extradata):
        return '{}s'.format(fileobj.payload_header.encryption_key_length)

    def __str__(self):
        return "<RoflEncryptionKey - key as hex: {}>".format(
            self.encryption_key.encode('hex')
        )

    fields = ['encryption_key']

    def unpack_encryption_key(self, field_name, field_value, *args, **kwargs):
        self.encryption_key = field_value.decode('base64')

    def pack_encryption_key(self, field_name, *args, **kwargs):
        return self.encryption_key.encode('base64')


class RoflChunkHeader(Struct):
    format = '<IBIII'
    fields = ['id', 'type', 'length', 'next_chunk_id', 'offset']

    def __str__(self):
        return "<RoflChunkHeader - id: {}, type: {}, length: {}>".format(
            self.id, 'CHUNK' if self.type == 1 else 'KEYFRAME', self.length
        )


class RoflChunkPayload(Struct):
    @classmethod
    def get_format(cls, outer, extradata):
        return '{}s'.format(extradata)

    def unpack_chunk(self, field_name, field_value, outer, extradata):
        self.length = len(field_value)
        if False:
            self.chunk = field_value
        else:
            crypto = outer.outer.crypto
            # we need to decrypt this
            chunk = StringIO(crypto.decrypt(crypto.keyframe_key, field_value))
            # now decompress
            chunkgz = gzip.GzipFile(fileobj=chunk)
            self.chunk = chunkgz.read()

    fields = ['chunk']

    def __str__(self):
        return "<RoflChunkPayload - length: {}>".format(self.length)


class RoflChunkHeaders(CompositeStructList):
    @classmethod
    def get_format(cls, fileobj, extradata):
        return [RoflChunkHeader] * fileobj.payload_header.chunk_count


class RoflKeyframeHeaders(CompositeStructList):
    @classmethod
    def get_format(cls, fileobj, extradata):
        return [RoflChunkHeader] * fileobj.payload_header.keyframe_count


class RoflChunks(CompositeStructList):
    @classmethod
    def get_format(cls, fileobj, extradata):
        return [RoflChunkPayload] * (
            len(fileobj.chunk_headers.data) +
            len(fileobj.keyframe_headers.data)
        )

    @classmethod
    def get_extradata(cls, fileobj):
        out = []
        for h in (fileobj.chunk_headers.data + fileobj.keyframe_headers.data):
            out.append(h.length)
        return out


class RoflCrypto(object):
    def __init__(self, roflfile):
        self.file = roflfile

    def make_crypto(self, key):
        from Crypto.Cipher import Blowfish
        return Blowfish.new(key, Blowfish.MODE_ECB)

    def decrypt(self, key, data):
        crypto = self.make_crypto(key)
        datawpadding = crypto.decrypt(data)
        # how much padding do we need to remove?
        paddingbytes = ord(datawpadding[-1])
        return datawpadding[:-paddingbytes]

    @property
    def keyframe_key(self):
        key = self.decrypt(
            str(self.file.metadata.json['gameId']),
            self.file.encryption_key.encryption_key
        )
        return key


class RoflFile(object):
    @classmethod
    def read(cls, fh):
        self = cls()
        self.header = RoflHeader.read(fh, self)
        if self.header.magic != ROFL_MAGIC:
            raise Exception("Decoding error - magic invalid")
        self.metadata = RoflMetadata.read(fh, self)
        self.payload_header = RoflPayloadHeader.read(fh, self)
        self.encryption_key = RoflEncryptionKey.read(fh, self)
        self.chunk_headers = RoflChunkHeaders.read(fh, self)
        self.keyframe_headers = RoflKeyframeHeaders.read(fh, self)
        self.chunks = RoflChunks.read(fh, self)

        self.chunk_pairs = zip(
            self.chunk_headers.data + self.keyframe_headers.data,
            self.chunks.data
        )
        return self

    @property
    def crypto(self):
        return RoflCrypto(self)

    def __str__(self):
        return "<RoflFile - header: {} - metadata: {} - payload header: {}" + \
            " - encryption key: {}>".format(
                self.header, self.metadata,
                self.payload_header, self.encryption_key
            )


def unpack_rofl_to_directory(rofl_file, directory):
    import os

    keyframe_dir = os.path.join(directory, 'keyframe')
    chunk_dir = os.path.join(directory, 'chunk')
    os.makedirs(keyframe_dir)
    os.makedirs(chunk_dir)

    with open(rofl_file, 'rb') as f:
        roflfile = RoflFile.read(f)

    for header, chunk in roflfile.chunk_pairs:
        print header, chunk
        base_dir = keyframe_dir if header.type == 2 else chunk_dir
        with open(os.path.join(base_dir, str(header.id)), 'wb') as c:
            c.write(chunk.chunk)
    with open(os.path.join(directory, 'meta.json'), 'wb') as f:
        roflfile.metadata.json['gameKey'] = {
            'gameId': roflfile.metadata.json['gameId']
        }
        roflfile.metadata.json['key'] = roflfile.encryption_key.\
            encryption_key.encode('base64')
        f.write(roflfile.metadata.as_json())


if __name__ == '__main__':
    import sys
    import os.path

    if len(sys.argv) != 3:
        print "{} <ROFL file> <directory>".format(sys.argv[0])
        sys.exit(0)

    my_name, rofl_file, output_dir = sys.argv

    if not os.path.exists(rofl_file):
        print "ROFL file specified does not exist"
        sys.exit(1)

    if os.path.exists(output_dir):
        print "Output directory exists - won't unpack"
        sys.exit(1)

unpack_rofl_to_directory(rofl_file, output_dir)


'''
