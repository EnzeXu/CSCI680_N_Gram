public class AESTest { private Random mRand = new Random(); @Test public void testEncrypt() throws InvalidKeyException, NoSuchAlgorithmException, NoSuchPaddingException, IllegalBlockSizeException, BadPaddingException, InvalidAlgorithmParameterException { testFinal(15); testFinal(16); testFinal(17); int size = mRand.nextInt(494) + 18; testFinal(size); } private void testFinal(int dataSize) throws NoSuchAlgorithmException, NoSuchPaddingException, IllegalBlockSizeException, BadPaddingException, InvalidKeyException, InvalidAlgorithmParameterException { byte[] input = new byte[dataSize]; mRand.nextBytes(input); byte[] keyArray = new byte[32]; mRand.nextBytes(keyArray); SecretKeySpec key = new SecretKeySpec(keyArray, "AES"); byte[] ivArray = new byte[16]; mRand.nextBytes(ivArray); IvParameterSpec iv = new IvParameterSpec(ivArray); Cipher android = CipherFactory.getInstance("AES/CBC/PKCS5Padding", true); android.init(Cipher.ENCRYPT_MODE, key, iv); byte[] outAndroid = android.doFinal(input, 0, dataSize); Cipher nat = CipherFactory.getInstance("AES/CBC/PKCS5Padding"); nat.init(Cipher.ENCRYPT_MODE, key, iv); byte[] outNative = nat.doFinal(input, 0, dataSize); assertArrayEquals("Arrays differ on size: " + dataSize, outAndroid, outNative); } }