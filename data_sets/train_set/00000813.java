public class AndroidFinalKey extends FinalKey { @SuppressLint("GetInstance") @Override public byte[] transformMasterKey(byte[] pKeySeed, byte[] pKey, long rounds) throws IOException { Cipher cipher; try { cipher = Cipher.getInstance("AES/ECB/NoPadding"); } catch (NoSuchAlgorithmException e) { throw new IOException("NoSuchAlgorithm: " + e.getMessage()); } catch (NoSuchPaddingException e) { throw new IOException("NoSuchPadding: " + e.getMessage()); } try { cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(pKeySeed, "AES")); } catch (InvalidKeyException e) { throw new IOException("InvalidPasswordException: " + e.getMessage()); } byte[] newKey = new byte[pKey.length]; System.arraycopy(pKey, 0, newKey, 0, pKey.length); byte[] destKey = new byte[pKey.length]; for (int i = 0; i < rounds; i++) { try { cipher.update(newKey, 0, newKey.length, destKey, 0); System.arraycopy(destKey, 0, newKey, 0, newKey.length); } catch (ShortBufferException e) { throw new IOException("Short buffer: " + e.getMessage()); } } MessageDigest md = null; try { md = MessageDigest.getInstance("SHA-256"); } catch (NoSuchAlgorithmException e) { assert true; throw new IOException("SHA-256 not implemented here: " + e.getMessage()); } md.update(newKey); return md.digest(); } }