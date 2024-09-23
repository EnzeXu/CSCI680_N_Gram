public class KeyGenParameterSpecCompat { private static Class builder ; private static Constructor buildConst ; private static Method builderBuild ; private static Method setBlockModes ; private static Method setUserAuthReq ; private static Method setEncPad ; private static boolean available ; static { try { builder = Class . forName ( "android . security . keystore . KeyGenParameterSpec$Builder" ) ; buildConst = builder . getConstructor ( String . class , int . class ) ; builderBuild = builder . getMethod ( "build" , ( Class [ ] ) null ) ; setBlockModes = builder . getMethod ( "setBlockModes" , String [ ] . class ) ; setUserAuthReq = builder . getMethod ( "setUserAuthenticationRequired" , new Class [ ] { boolean . class } ) ; setEncPad = builder . getMethod ( "setEncryptionPaddings" , String [ ] . class ) ; available = true ; } catch ( Exception e ) { available = false ; } } public static AlgorithmParameterSpec build ( String keystoreAlias , int purpose , String blockMode , boolean userAuthReq , String encPadding ) { if ( !available ) { return null ; } try { Object inst = buildConst . newInstance ( keystoreAlias , purpose ) ; inst = setBlockModes . invoke ( inst , new Object [ ] { new String [ ] { blockMode } } ) ; inst = setUserAuthReq . invoke ( inst , userAuthReq ) ; inst = setEncPad . invoke ( inst , new Object [ ] { new String [ ] { encPadding } } ) ; return ( AlgorithmParameterSpec ) builderBuild . invoke ( inst , null ) ; } catch ( Exception e ) { return null ; } } }