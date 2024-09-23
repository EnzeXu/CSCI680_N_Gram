public class S3Rule extends ExternalResource { public static final String IDENTITY = "identity"; public static final String CREDENTIAL = "credential"; public static final String REGION = "us-east-1"; public static final String BUCKET = "bucket"; public static String DEFAULT_URL = "http: Properties properties = new Properties(); S3Proxy s3Proxy; AmazonS3 s3client; String endpoint = DEFAULT_URL; String bucketName = BUCKET; String directoryKeyPrefix = null; String inputPath; Supplier<String> storageDir; public S3Rule( Supplier<String> storageDir, String bucketName, String inputPath ) { this( storageDir ); this.bucketName = bucketName; this.inputPath = inputPath; } public S3Rule( Supplier<String> storageDir, String bucketName ) { this( storageDir ); this.bucketName = bucketName; } public S3Rule( Supplier<String> storageDir ) { this.storageDir = storageDir; } public S3Rule( String storageDir ) { this( () -> storageDir ); } public S3Rule( String storageDir, String bucketName ) { this( () -> storageDir ); this.bucketName = bucketName; } public AmazonS3 get3Client() { if( s3client != null ) return s3client; s3client = AmazonS3ClientBuilder.standard() .enablePathStyleAccess() .withClientConfiguration( new ClientConfiguration().withSignerOverride( "S3SignerType" ) ) .withCredentials( new AWSStaticCredentialsProvider( new BasicAWSCredentials( IDENTITY, CREDENTIAL ) ) ) .withEndpointConfiguration( new AwsClientBuilder.EndpointConfiguration( endpoint, REGION ) ).build(); return s3client; } public String getEndpoint() { return endpoint; } @Override protected void before() throws Throwable { properties.setProperty( "jclouds.filesystem.basedir", storageDir.get() ); BlobStoreContext context = ContextBuilder .newBuilder( "filesystem" ) .credentials( IDENTITY, CREDENTIAL ) .overrides( properties ) .build( BlobStoreContext.class ); s3Proxy = S3Proxy.builder() .blobStore( context.getBlobStore() ) .endpoint( URI.create( endpoint ) ) .build(); s3Proxy.start(); uploadData( inputPath ); } public void uploadData( String inputPath ) throws InterruptedException { if( inputPath == null ) return; createBucket( bucketName ); TransferManager transferManager = TransferManagerBuilder.standard().withS3Client( get3Client() ).build(); MultipleFileUpload upload = transferManager. uploadDirectory( bucketName, directoryKeyPrefix, new File( inputPath ), true ); upload.waitForCompletion(); } public void createBucket( String bucketName ) { AmazonS3 client = get3Client(); if( client.doesBucketExist( bucketName ) ) return; client.createBucket( bucketName ); } @Override protected void after() { try { if( s3Proxy != null ) s3Proxy.stop(); } catch( Exception exception ) { } } }