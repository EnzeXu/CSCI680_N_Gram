public class KdfParameters extends VariantDictionary { public UUID kdfUUID; private static final String ParamUUID = "$UUID"; public KdfParameters(UUID uuid) { kdfUUID = uuid; this.setByteArray(ParamUUID, Types.UUIDtoBytes(uuid)); } public static KdfParameters deserialize(byte[] data) throws IOException { ByteArrayInputStream bis = new ByteArrayInputStream(data); LEDataInputStream lis = new LEDataInputStream(bis); VariantDictionary d = VariantDictionary.deserialize(lis); if (d == null) { assert(false); return null; } byte[] uuidBytes = d.getByteArray((ParamUUID)); UUID uuid; if (uuidBytes != null) { uuid = Types.bytestoUUID(uuidBytes); } else { uuid = AesKdf.CIPHER_UUID; } if (uuid == null) { assert(false); return null; } KdfParameters kdfP = new KdfParameters(uuid); kdfP.copyTo(d); return kdfP; } public static byte[] serialize(KdfParameters kdf) throws IOException { ByteArrayOutputStream bos = new ByteArrayOutputStream(); LEDataOutputStream los = new LEDataOutputStream(bos); KdfParameters.serialize(kdf, los); return bos.toByteArray(); } }