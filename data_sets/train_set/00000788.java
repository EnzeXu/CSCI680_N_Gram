public class PwDbV3Output extends PwDbOutput { private PwDatabaseV3 mPM; private byte[] headerHashBlock; public PwDbV3Output(PwDatabaseV3 pm, OutputStream os) { super(os); mPM = pm; } public byte[] getFinalKey(PwDbHeader header) throws PwDbOutputException { try { PwDbHeaderV3 h3 = (PwDbHeaderV3) header; mPM.makeFinalKey(h3.masterSeed, h3.transformSeed, mPM.numKeyEncRounds); return mPM.finalKey; } catch (IOException e) { throw new PwDbOutputException("Key creation failed: " + e.getMessage()); } } @Override public void output() throws PwDbOutputException { prepForOutput(); PwDbHeader header = outputHeader(mOS); byte[] finalKey = getFinalKey(header); Cipher cipher; try { if (mPM.algorithm == PwEncryptionAlgorithm.Rjindal) { cipher = CipherFactory.getInstance("AES/CBC/PKCS5Padding"); } else if (mPM.algorithm == PwEncryptionAlgorithm.Twofish){ cipher = CipherFactory.getInstance("Twofish/CBC/PKCS7PADDING"); } else { throw new Exception(); } } catch (Exception e) { throw new PwDbOutputException("Algorithm not supported."); } try { cipher.init( Cipher.ENCRYPT_MODE, new SecretKeySpec(finalKey, "AES" ), new IvParameterSpec(header.encryptionIV) ); CipherOutputStream cos = new CipherOutputStream(mOS, cipher); BufferedOutputStream bos = new BufferedOutputStream(cos); outputPlanGroupAndEntries(bos); bos.flush(); bos.close(); } catch (InvalidKeyException e) { throw new PwDbOutputException("Invalid key"); } catch (InvalidAlgorithmParameterException e) { throw new PwDbOutputException("Invalid algorithm parameter."); } catch (IOException e) { throw new PwDbOutputException("Failed to output final encrypted part."); } } private void prepForOutput() { sortGroupsForOutput(); } @Override protected SecureRandom setIVs(PwDbHeader header) throws PwDbOutputException { SecureRandom random = super.setIVs(header); PwDbHeaderV3 h3 = (PwDbHeaderV3) header; random.nextBytes(h3.transformSeed); return random; } public PwDbHeaderV3 outputHeader(OutputStream os) throws PwDbOutputException { PwDbHeaderV3 header = new PwDbHeaderV3(); header.signature1 = PwDbHeader.PWM_DBSIG_1; header.signature2 = PwDbHeaderV3.DBSIG_2; header.flags = PwDbHeaderV3.FLAG_SHA2; if ( mPM.getEncAlgorithm() == PwEncryptionAlgorithm.Rjindal ) { header.flags |= PwDbHeaderV3.FLAG_RIJNDAEL; } else if ( mPM.getEncAlgorithm() == PwEncryptionAlgorithm.Twofish ) { header.flags |= PwDbHeaderV3.FLAG_TWOFISH; } else { throw new PwDbOutputException("Unsupported algorithm."); } header.version = PwDbHeaderV3.DBVER_DW; header.numGroups = mPM.getGroups().size(); header.numEntries = mPM.entries.size(); header.numKeyEncRounds = mPM.getNumKeyEncRecords(); setIVs(header); MessageDigest md = null; try { md = MessageDigest.getInstance("SHA-256"); } catch (NoSuchAlgorithmException e) { throw new PwDbOutputException("SHA-256 not implemented here."); } MessageDigest headerDigest; try { headerDigest = MessageDigest.getInstance("SHA-256"); } catch (NoSuchAlgorithmException e) { throw new PwDbOutputException("SHA-256 not implemented here."); } NullOutputStream nos; nos = new NullOutputStream(); DigestOutputStream headerDos = new DigestOutputStream(nos, headerDigest); PwDbHeaderOutputV3 pho = new PwDbHeaderOutputV3(header, headerDos); try { pho.outputStart(); pho.outputEnd(); headerDos.flush(); } catch (IOException e) { throw new PwDbOutputException(e); } byte[] headerHash = headerDigest.digest(); headerHashBlock = getHeaderHashBuffer(headerHash); nos = new NullOutputStream(); DigestOutputStream dos = new DigestOutputStream(nos, md); BufferedOutputStream bos = new BufferedOutputStream(dos); try { outputPlanGroupAndEntries(bos); bos.flush(); bos.close(); } catch (IOException e) { throw new PwDbOutputException("Failed to generate checksum."); } header.contentsHash = md.digest(); pho = new PwDbHeaderOutputV3(header, os); try { pho.outputStart(); dos.on(false); pho.outputContentHash(); dos.on(true); pho.outputEnd(); dos.flush(); } catch (IOException e) { throw new PwDbOutputException(e); } return header; } public void outputPlanGroupAndEntries(OutputStream os) throws PwDbOutputException { LEDataOutputStream los = new LEDataOutputStream(os); if (useHeaderHash() && headerHashBlock != null) { try { los.writeUShort(0x0000); los.writeInt(headerHashBlock.length); los.write(headerHashBlock); } catch (IOException e) { throw new PwDbOutputException("Failed to output header hash: " + e.getMessage()); } } List<PwGroup> groups = mPM.getGroups(); for ( int i = 0; i < groups.size(); i++ ) { PwGroupV3 pg = (PwGroupV3) groups.get(i); PwGroupOutputV3 pgo = new PwGroupOutputV3(pg, os); try { pgo.output(); } catch (IOException e) { throw new PwDbOutputException("Failed to output a group: " + e.getMessage()); } } for (int i = 0; i < mPM.entries.size(); i++ ) { PwEntryV3 pe = (PwEntryV3) mPM.entries.get(i); PwEntryOutputV3 peo = new PwEntryOutputV3(pe, os); try { peo.output(); } catch (IOException e) { throw new PwDbOutputException("Failed to output an entry."); } } } private void sortGroupsForOutput() { List<PwGroup> groupList = new ArrayList<PwGroup>(); List<PwGroup> roots = mPM.getGrpRoots(); for ( int i = 0; i < roots.size(); i++ ) { sortGroup((PwGroupV3) roots.get(i), groupList); } mPM.setGroups(groupList); } private void sortGroup(PwGroupV3 group, List<PwGroup> groupList) { groupList.add(group); for ( int i = 0; i < group.childGroups.size(); i++ ) { sortGroup((PwGroupV3) group.childGroups.get(i), groupList); } } private byte[] getHeaderHashBuffer(byte[] headerDigest) { ByteArrayOutputStream baos = new ByteArrayOutputStream(); try { writeExtData(headerDigest, baos); return baos.toByteArray(); } catch (IOException e) { return null; } } private void writeExtData(byte[] headerDigest, OutputStream os) throws IOException { LEDataOutputStream los = new LEDataOutputStream(os); writeExtDataField(los, 0x0001, headerDigest, headerDigest.length); byte[] headerRandom = new byte[32]; SecureRandom rand = new SecureRandom(); rand.nextBytes(headerRandom); writeExtDataField(los, 0x0002, headerRandom, headerRandom.length); writeExtDataField(los, 0xFFFF, null, 0); } private void writeExtDataField(LEDataOutputStream los, int fieldType, byte[] data, int fieldSize) throws IOException { los.writeUShort(fieldType); los.writeInt(fieldSize); if (data != null) { los.write(data); } } protected boolean useHeaderHash() { return true; } }