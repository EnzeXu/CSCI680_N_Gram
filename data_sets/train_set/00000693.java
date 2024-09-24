public class ProtectedBinaryTest { @Test public void testEncryption() throws Exception { byte[] input = new byte[4096]; Random random = new Random(); random.nextBytes(input); Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext(); File dir = ctx.getFilesDir(); File temp = new File(dir, "1"); ProtectedBinary pb = new ProtectedBinary(true, temp, input.length); OutputStream os = pb.getOutputStream(); ByteArrayInputStream bais = new ByteArrayInputStream(input); Util.copyStream(bais, os); os.close(); InputStream is = pb.getData(); ByteArrayOutputStream baos = new ByteArrayOutputStream(); Util.copyStream(is, baos); byte[] output = baos.toByteArray(); assertArrayEquals(input, output); } }