public class TwofishEngine extends CipherEngine { public static final UUID CIPHER_UUID = Types.bytestoUUID( new byte[]{(byte)0xAD, (byte)0x68, (byte)0xF2, (byte)0x9F, (byte)0x57, (byte)0x6F, (byte)0x4B, (byte)0xB9, (byte)0xA3, (byte)0x6A, (byte)0xD4, (byte)0x7A, (byte)0xF9, (byte)0x65, (byte)0x34, (byte)0x6C }); @Override public Cipher getCipher(int opmode, byte[] key, byte[] IV, boolean androidOverride) throws NoSuchAlgorithmException, NoSuchPaddingException, InvalidKeyException, InvalidAlgorithmParameterException { Cipher cipher; if (opmode == Cipher.ENCRYPT_MODE) { cipher = CipherFactory.getInstance("Twofish/CBC/ZeroBytePadding", androidOverride); } else { cipher = CipherFactory.getInstance("Twofish/CBC/NoPadding", androidOverride); } cipher.init(opmode, new SecretKeySpec(key, "AES"), new IvParameterSpec(IV)); return cipher; } }