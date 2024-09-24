public class PwStreamCipherFactory { public static StreamCipher getInstance(CrsAlgorithm alg, byte[] key) { if ( alg == CrsAlgorithm.Salsa20 ) { return getSalsa20(key); } else if (alg == CrsAlgorithm.ChaCha20) { return getChaCha20(key); } else { return null; } } private static final byte[] SALSA_IV = new byte[]{ (byte)0xE8, 0x30, 0x09, 0x4B, (byte)0x97, 0x20, 0x5D, 0x2A }; private static StreamCipher getSalsa20(byte[] key) { byte[] key32 = CryptoUtil.hashSha256(key); KeyParameter keyParam = new KeyParameter(key32); ParametersWithIV ivParam = new ParametersWithIV(keyParam, SALSA_IV); StreamCipher cipher = new Salsa20Engine(); cipher.init(true, ivParam); return cipher; } private static StreamCipher getChaCha20(byte[] key) { byte[] hash = CryptoUtil.hashSha512(key); byte[] key32 = new byte[32]; byte[] iv = new byte[12]; System.arraycopy(hash, 0, key32, 0, 32); System.arraycopy(hash, 32, iv, 0, 12); KeyParameter keyParam = new KeyParameter(key32); ParametersWithIV ivParam = new ParametersWithIV(keyParam, iv); StreamCipher cipher = new ChaCha7539Engine(); cipher.init(true, ivParam); return cipher; } }