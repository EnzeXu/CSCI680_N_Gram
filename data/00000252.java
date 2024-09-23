public class RFont implements PConstants { Font f ; float scaleFactor = 0 . 2F ; public int size = DEFAULT_SIZE ; public int align = DEFAULT_ALIGN ; final static int DEFAULT_SIZE = 48 ; final static int DEFAULT_RESOLUTION = 72 ; final static int DEFAULT_ALIGN = RFont . LEFT ; public boolean forceAscii = false ; public RFont ( String fontPath , int size , int align ) throws RuntimeException { byte [ ] bs = RG . parent ( ) . loadBytes ( fontPath ) ; f = Font . create ( bs ) ; setSize ( size ) ; setAlign ( align ) ; } public RFont ( String fontPath , int size ) throws RuntimeException { this ( fontPath , size , DEFAULT_ALIGN ) ; } public RFont ( String fontPath ) throws RuntimeException { this ( fontPath , DEFAULT_SIZE , DEFAULT_ALIGN ) ; } public void setSize ( int size ) { short unitsPerEm = f . getHeadTable ( ) . getUnitsPerEm ( ) ; int resolution = RG . dpi ( ) ; this . scaleFactor = ( ( float ) size * ( float ) resolution ) / ( 72F * ( float ) unitsPerEm ) ; } public float getLineSpacing ( ) { short unitsPerEm = f . getHeadTable ( ) . getUnitsPerEm ( ) ; System . out . println ( "UnitsPerEm ( emsize ) : " + unitsPerEm ) ; float hheaLineGap = ( f . getHheaTable ( ) . getAscender ( ) - f . getHheaTable ( ) . getDescender ( ) + f . getHheaTable ( ) . getLineGap ( ) ) * this . scaleFactor ; System . out . println ( "HHEA lineGap : " + hheaLineGap ) ; float os2TypoLineGap = ( f . getOS2Table ( ) . getTypoAscender ( ) - f . getOS2Table ( ) . getTypoDescender ( ) + f . getOS2Table ( ) . getTypoLineGap ( ) ) * this . scaleFactor ; System . out . println ( "Os2 Typo lineGap : " + os2TypoLineGap ) ; float os2WinLineGap = ( f . getOS2Table ( ) . getWinAscent ( ) + f . getOS2Table ( ) . getWinDescent ( ) ) * this . scaleFactor ; System . out . println ( "Os2 Win lineGap : " + os2WinLineGap ) ; float autoLineGap = f . getHeadTable ( ) . getUnitsPerEm ( ) * 1 . 25f * this . scaleFactor ; System . out . println ( "Automatic lineGap : " + autoLineGap ) ; return hheaLineGap ; } public void setAlign ( int align ) throws RuntimeException { if ( align!=LEFT && align!=CENTER && align!=RIGHT ) { throw new RuntimeException ( "Alignment unknown . The only accepted values are : RFont . LEFT , RFont . CENTER and RFont . RIGHT" ) ; } this . align = align ; } public String getFamily ( ) { return f . getNameTable ( ) . getRecord ( org . apache . batik . svggen . font . table . Table . nameFontFamilyName ) ; } public RShape toShape ( char character ) { RGroup grp = toGroup ( Character . toString ( character ) ) ; if ( grp . countElements ( ) > 0 ) return ( RShape ) ( grp . elements [ 0 ] ) ; return new RShape ( ) ; } public RPolygon toPolygon ( char character ) { return toShape ( character ) . toPolygon ( ) ; } private CmapFormat getCmapFormat ( ) { if ( forceAscii ) { return f . getCmapTable ( ) . getCmapFormat ( org . apache . batik . svggen . font . table . Table . platformMacintosh , org . apache . batik . svggen . font . table . Table . encodingRoman ) ; } else { short [ ] platforms = new short [ ] { org . apache . batik . svggen . font . table . Table . platformMicrosoft , org . apache . batik . svggen . font . table . Table . platformAppleUnicode , org . apache . batik . svggen . font . table . Table . platformMacintosh } ; short [ ] encodings = new short [ ] { org . apache . batik . svggen . font . table . Table . encodingUGL , org . apache . batik . svggen . font . table . Table . encodingKorean , org . apache . batik . svggen . font . table . Table . encodingHebrew , org . apache . batik . svggen . font . table . Table . encodingUndefined } ; CmapFormat cmapFmt ; for ( int i = 0 ; i < encodings . length ; i++ ) { for ( int j = 0 ; j < platforms . length ; j++ ) { cmapFmt = f . getCmapTable ( ) . getCmapFormat ( platforms [ j ] , encodings [ i ] ) ; if ( cmapFmt != null ) { return cmapFmt ; } } } return null ; } } public RGroup toGroup ( String text ) throws RuntimeException { RGroup result = new RGroup ( ) ; CmapFormat cmapFmt = getCmapFormat ( ) ; if ( cmapFmt == null ) { throw new RuntimeException ( "Cannot find a suitable cmap table" ) ; } int x = 0 ; for ( short i = 0 ; i < text . length ( ) ; i++ ) { int glyphIndex = cmapFmt . mapCharCode ( text . charAt ( i ) ) ; Glyph glyph = f . getGlyph ( glyphIndex ) ; int default_advance_x = f . getHmtxTable ( ) . getAdvanceWidth ( glyphIndex ) ; if ( glyph != null ) { glyph . scale ( scaleFactor ) ; result . addElement ( getGlyphAsShape ( f , glyph , glyphIndex , x ) ) ; x += glyph . getAdvanceWidth ( ) ; } else { x += ( int ) ( ( float ) default_advance_x*scaleFactor ) ; } } if ( align!=LEFT && align!=CENTER && align!=RIGHT ) { throw new RuntimeException ( "Alignment unknown . The only accepted values are : RFont . LEFT , RFont . CENTER and RFont . RIGHT" ) ; } RRectangle r ; RMatrix mattrans ; switch ( this . align ) { case RFont . CENTER : r = result . getBounds ( ) ; mattrans = new RMatrix ( ) ; mattrans . translate ( ( r . getMinX ( ) -r . getMaxX ( ) ) /2 , 0 ) ; result . transform ( mattrans ) ; break ; case RFont . RIGHT : r = result . getBounds ( ) ; mattrans = new RMatrix ( ) ; mattrans . translate ( ( r . getMinX ( ) -r . getMaxX ( ) ) , 0 ) ; result . transform ( mattrans ) ; break ; case RFont . LEFT : break ; } return result ; } public RShape toShape ( String text ) throws RuntimeException { RShape result = new RShape ( ) ; CmapFormat cmapFmt = getCmapFormat ( ) ; if ( cmapFmt == null ) { throw new RuntimeException ( "Cannot find a suitable cmap table" ) ; } int x = 0 ; for ( short i = 0 ; i < text . length ( ) ; i++ ) { int glyphIndex = cmapFmt . mapCharCode ( text . charAt ( i ) ) ; Glyph glyph = f . getGlyph ( glyphIndex ) ; int default_advance_x = f . getHmtxTable ( ) . getAdvanceWidth ( glyphIndex ) ; if ( glyph != null ) { glyph . scale ( scaleFactor ) ; result . addChild ( getGlyphAsShape ( f , glyph , glyphIndex , x ) ) ; x += glyph . getAdvanceWidth ( ) ; } else { x += ( int ) ( ( float ) default_advance_x*scaleFactor ) ; } } if ( align!=LEFT && align!=CENTER && align!=RIGHT ) { throw new RuntimeException ( "Alignment unknown . The only accepted values are : RFont . LEFT , RFont . CENTER and RFont . RIGHT" ) ; } RRectangle r ; RMatrix mattrans ; switch ( this . align ) { case RFont . CENTER : r = result . getBounds ( ) ; mattrans = new RMatrix ( ) ; mattrans . translate ( ( r . getMinX ( ) -r . getMaxX ( ) ) /2 , 0 ) ; result . transform ( mattrans ) ; break ; case RFont . RIGHT : r = result . getBounds ( ) ; mattrans = new RMatrix ( ) ; mattrans . translate ( ( r . getMinX ( ) -r . getMaxX ( ) ) , 0 ) ; result . transform ( mattrans ) ; break ; case RFont . LEFT : break ; } return result ; } public void draw ( char character , PGraphics g ) throws RuntimeException { this . toShape ( character ) . draw ( g ) ; } public void draw ( String text , PGraphics g ) throws RuntimeException { this . toGroup ( text ) . draw ( g ) ; } public void draw ( char character , PApplet g ) throws RuntimeException { this . toShape ( character ) . draw ( g ) ; } public void draw ( String text , PApplet g ) throws RuntimeException { this . toGroup ( text ) . draw ( g ) ; } public void draw ( String text ) throws RuntimeException { this . toGroup ( text ) . draw ( ) ; } public void draw ( char character ) throws RuntimeException { this . toShape ( character ) . draw ( ) ; } private static float midValue ( float a , float b ) { return a + ( b - a ) /2 ; } protected static RShape getContourAsShape ( Glyph glyph , int startIndex , int count ) { return getContourAsShape ( glyph , startIndex , count , 0 ) ; } protected static RShape getContourAsShape ( Glyph glyph , int startIndex , int count , float xadv ) { if ( glyph . getPoint ( startIndex ) . endOfContour ) { return new RShape ( ) ; } RShape result = new RShape ( ) ; int offset = 0 ; while ( offset < count ) { Point point = glyph . getPoint ( startIndex + offset%count ) ; Point point_plus1 = glyph . getPoint ( startIndex + ( offset+1 ) %count ) ; Point point_plus2 = glyph . getPoint ( startIndex + ( offset+2 ) %count ) ; float pointx = ( ( float ) point . x + xadv ) ; float pointy = ( ( float ) point . y ) ; float point_plus1x = ( ( float ) point_plus1 . x + xadv ) ; float point_plus1y = ( ( float ) point_plus1 . y ) ; float point_plus2x = ( ( float ) point_plus2 . x + xadv ) ; float point_plus2y = ( ( float ) point_plus2 . y ) ; if ( offset == 0 ) { result . addMoveTo ( pointx , pointy ) ; } if ( point . onCurve && point_plus1 . onCurve ) { result . addLineTo ( point_plus1x , point_plus1y ) ; offset++ ; } else if ( point . onCurve && !point_plus1 . onCurve && point_plus2 . onCurve ) { result . addQuadTo ( point_plus1x , point_plus1y , point_plus2x , point_plus2y ) ; offset+=2 ; } else if ( point . onCurve && !point_plus1 . onCurve && !point_plus2 . onCurve ) { result . addQuadTo ( point_plus1x , point_plus1y , midValue ( point_plus1x , point_plus2x ) , midValue ( point_plus1y , point_plus2y ) ) ; offset+=2 ; } else if ( !point . onCurve && !point_plus1 . onCurve ) { result . addQuadTo ( pointx , pointy , midValue ( pointx , point_plus1x ) , midValue ( pointy , point_plus1y ) ) ; offset++ ; } else if ( !point . onCurve && point_plus1 . onCurve ) { result . addQuadTo ( pointx , pointy , point_plus1x , point_plus1y ) ; offset++ ; } else { System . out . println ( "drawGlyph case not catered for!!" ) ; break ; } } result . addClose ( ) ; return result ; } protected static RShape getGlyphAsShape ( Font font , Glyph glyph , int glyphIndex ) { return getGlyphAsShape ( font , glyph , glyphIndex , 0 ) ; } protected static RShape getGlyphAsShape ( Font font , Glyph glyph , int glyphIndex , float xadv ) { RShape result = new RShape ( ) ; int firstIndex = 0 ; int count = 0 ; int i ; if ( glyph != null ) { for ( i = 0 ; i < glyph . getPointCount ( ) ; i++ ) { count++ ; if ( glyph . getPoint ( i ) . endOfContour ) { result . addShape ( getContourAsShape ( glyph , firstIndex , count , xadv ) ) ; firstIndex = i + 1 ; count = 0 ; } } } return result ; } protected static RShape getGlyphAsShape ( Font font , Glyph glyph , int glyphIndex , SingleSubst arabInitSubst , SingleSubst arabMediSubst , SingleSubst arabTermSubst ) { return getGlyphAsShape ( font , glyph , glyphIndex , arabInitSubst , arabMediSubst , arabTermSubst , 0 ) ; } protected static RShape getGlyphAsShape ( Font font , Glyph glyph , int glyphIndex , SingleSubst arabInitSubst , SingleSubst arabMediSubst , SingleSubst arabTermSubst , float xadv ) { RShape result = new RShape ( ) ; boolean substituted = false ; int arabInitGlyphIndex = glyphIndex ; int arabMediGlyphIndex = glyphIndex ; int arabTermGlyphIndex = glyphIndex ; if ( arabInitSubst != null ) { arabInitGlyphIndex = arabInitSubst . substitute ( glyphIndex ) ; } if ( arabMediSubst != null ) { arabMediGlyphIndex = arabMediSubst . substitute ( glyphIndex ) ; } if ( arabTermSubst != null ) { arabTermGlyphIndex = arabTermSubst . substitute ( glyphIndex ) ; } if ( arabInitGlyphIndex != glyphIndex ) { result . addShape ( getGlyphAsShape ( font , font . getGlyph ( arabInitGlyphIndex ) , arabInitGlyphIndex ) ) ; substituted = true ; } if ( arabMediGlyphIndex != glyphIndex ) { result . addShape ( getGlyphAsShape ( font , font . getGlyph ( arabMediGlyphIndex ) , arabMediGlyphIndex ) ) ; substituted = true ; } if ( arabTermGlyphIndex != glyphIndex ) { result . addShape ( getGlyphAsShape ( font , font . getGlyph ( arabTermGlyphIndex ) , arabTermGlyphIndex ) ) ; substituted = true ; } if ( substituted ) { result . addShape ( getGlyphAsShape ( font , glyph , glyphIndex ) ) ; } else { result . addShape ( getGlyphAsShape ( font , glyph , glyphIndex ) ) ; } return result ; } }