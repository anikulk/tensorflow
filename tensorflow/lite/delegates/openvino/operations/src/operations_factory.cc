class OperationsBase {
 protected:
  enum ConversionType {
    NHWC_NCHW,
    NCHW_NHWC,
    IHWO_OIHW,
    OHWI_OIHW,
    NHWC_CWHN,
    CWHN_NHWC,
    NHC_NCH,
    NCH_NHC,
    CNH_NHC,
    NCH_HNC,
    HNC_NCH,
    NHC_CNH,
    BTS_TBS,
    NHCW_NHWC,
    NC_CN
  };

  std::shared_ptr<ov::Node> transpose(ConversionType type,
                                      ov::Output<ov::Node> input);
  virtual std::shared_ptr<ov::Node> createNode() = 0;
  // override createNodeForPlugin in case sPluginType specific implementation is
  // required
  virtual std::shared_ptr<ov::Node> createNodeForPlugin();
}